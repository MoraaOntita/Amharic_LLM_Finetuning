import re
import os
import io
import nltk
from nltk.collocations import BigramCollocationFinder

class AmharicPreprocessor:
    def __init__(self, expansion_file_dir, bigram_dir):
        self.expansion_file_dir = expansion_file_dir
        self.bigram_dir = bigram_dir
        self.short_form_dict = self.get_short_forms()

    def get_short_forms(self):
        exp = {}
        try:
            with open(self.expansion_file_dir, encoding='utf8') as text:
                for line in text:
                    line = line.strip()
                    if not line:
                        continue
                    expanded = line.split("-")
                    exp[expanded[0].strip()] = expanded[1].replace(" ", '_').strip()
        except FileNotFoundError:
            print(f"File not found: {self.expansion_file_dir}")
        return exp

    def expand_short_form(self, input_short_word):
        return self.short_form_dict.get(input_short_word, input_short_word)

    def normalize_char_level_mismatch(self, input_token):
        replacements = [
            ('[ሃኅኃሐሓኻ]', 'ሀ'), ('[ሑኁዅ]', 'ሁ'), ('[ኂሒኺ]', 'ሂ'), ('[ኌሔዄ]', 'ሄ'), ('[ሕኅ]', 'ህ'),
            ('[ኆሖኾ]', 'ሆ'), ('[ሠ]', 'ሰ'), ('[ሡ]', 'ሱ'), ('[ሢ]', 'ሲ'), ('[ሣ]', 'ሳ'), ('[ሤ]', 'ሴ'),
            ('[ሥ]', 'ስ'), ('[ሦ]', 'ሶ'), ('[ዓኣዐ]', 'አ'), ('[ዑ]', 'ኡ'), ('[ዒ]', 'ኢ'), ('[ዔ]', 'ኤ'),
            ('[ዕ]', 'እ'), ('[ዖ]', 'ኦ'), ('[ጸ]', 'ፀ'), ('[ጹ]', 'ፁ'), ('[ጺ]', 'ፂ'), ('[ጻ]', 'ፃ'),
            ('[ጼ]', 'ፄ'), ('[ጽ]', 'ፅ'), ('[ጾ]', 'ፆ'), ('(ሉ[ዋአ])', 'ሏ'), ('(ሙ[ዋአ])', 'ሟ'),
            ('(ቱ[ዋአ])', 'ቷ'), ('(ሩ[ዋአ])', 'ሯ'), ('(ሱ[ዋአ])', 'ሷ'), ('(ሹ[ዋአ])', 'ሿ'),
            ('(ቁ[ዋአ])', 'ቋ'), ('(ቡ[ዋአ])', 'ቧ'), ('(ቹ[ዋአ])', 'ቿ'), ('(ሁ[ዋአ])', 'ኋ'),
            ('(ኑ[ዋአ])', 'ኗ'), ('(ኙ[ዋአ])', 'ኟ'), ('(ኩ[ዋአ])', 'ኳ'), ('(ዙ[ዋአ])', 'ዟ'),
            ('(ጉ[ዋአ])', 'ጓ'), ('(ደ[ዋአ])', 'ዷ'), ('(ጡ[ዋአ])', 'ጧ'), ('(ጩ[ዋአ])', 'ጯ'),
            ('(ጹ[ዋአ])', 'ጿ'), ('(ፉ[ዋአ])', 'ፏ'), ('[ቊ]', 'ቁ'), ('[ኵ]', 'ኩ')
        ]
        for pattern, replacement in replacements:
            input_token = re.sub(pattern, replacement, str(input_token))
        return input_token

    def remove_punc_and_special_chars(self, text):
        return re.sub(r'[\!\@\#\$\%\^\«\»\&\*\(\)\…\[\]\{\}\;\“\”\›\’\‘\"\'\:\,\.\‹\/\<\>\?\\\\|\`\´\~\-\=\+\፡\።\፤\;\፦\፥\፧\፨\፠\፣]', '', text)

    def remove_ascii_and_numbers(self, text_input):
        rm_num_and_ascii = re.sub('[A-Za-z0-9]', '', text_input)
        return re.sub('[\u1369-\u137C]+', '', rm_num_and_ascii)

    def arabic2geez(self, arabicNumber):
        ETHIOPIC_ONE = 0x1369
        ETHIOPIC_TEN = 0x1372
        ETHIOPIC_HUNDRED = 0x137B
        ETHIOPIC_TEN_THOUSAND = 0x137C

        arabicNumber = str(arabicNumber)
        n = len(arabicNumber) - 1
        if n % 2 == 0:
            arabicNumber = "0" + arabicNumber
            n += 1

        arabicBigrams = [arabicNumber[i:i+2] for i in range(0, n, 2)]
        reversedArabic = arabicBigrams[::-1]
        geez = []

        for index, pair in enumerate(reversedArabic):
            curr_geez = ''
            artens = pair[0]
            arones = pair[1]
            amtens = ''
            amones = ''
            if artens != '0':
                amtens = str(chr((int(artens) + (ETHIOPIC_TEN - 1))))
            else:
                if arones == '0':
                    continue
            if arones != '0':
                amones = str(chr((int(arones) + (ETHIOPIC_ONE - 1))))
            if index > 0:
                if index % 2 != 0:
                    curr_geez = amtens + amones + str(chr(ETHIOPIC_HUNDRED))
                else:
                    curr_geez = amtens + amones + str(chr(ETHIOPIC_TEN_THOUSAND))
            else:
                curr_geez = amtens + amones
            geez.append(curr_geez)

        geez = ''.join(geez[::-1])
        if geez.startswith('፩፻') or geez.startswith('፩፼'):
            geez = geez[1:]

        if len(arabicNumber) >= 7:
            end_zeros = ''.join(re.findall('([0]+)$', arabicNumber)[0:])
            i = int(len(end_zeros) / 3)
            if len(end_zeros) >= (3 * i):
                if i >= 3:
                    i -= 1
                for thoushand in range(i - 1):
                    geez += '፼'

        return geez

    def get_expanded_number(self, number):
        if '.' not in str(number):
            return self.arabic2geez(number)
        else:
            num, decimal = str(number).split('.')
            if decimal.startswith('0'):
                decimal = decimal[1:]
                dot = ' ነጥብ ዜሮ '
            else:
                dot = ' ነጥብ '
            return self.arabic2geez(num) + dot + self.arabic2geez(decimal)

    def tokenize(self, corpus):
        sentences = re.compile('[!?።\፡\፡]+').split(corpus)
        tokens = []
        for sentence in sentences:
            tokens.extend(sentence.split())
        return tokens

    def collocation_finder(self, tokens):
        bigram_measures = nltk.collocations.BigramAssocMeasures()
        finder = BigramCollocationFinder.from_words(tokens)
        finder.apply_freq_filter(3)
        frequent_bigrams = finder.nbest(bigram_measures.chi_sq, 5)
        
        with io.open(self.bigram_dir, "w", encoding="utf8") as PhraseWriter:
            for bigram in frequent_bigrams:
                PhraseWriter.write(bigram[0] + ' ' + bigram[1] + "\n")

    def normalize_multi_words(self, tokenized_sentence, corpus):
        bigram = set()
        sent_with_bigrams = []
        index = 0

        if not os.path.exists(self.bigram_dir):
            self.collocation_finder(self.tokenize(corpus))

        try:
            with open(self.bigram_dir, encoding='utf8') as phrase_file:
                for line in phrase_file:
                    bigram.add(tuple(line.strip().split()))
        except FileNotFoundError:
            print(f"File not found: {self.bigram_dir}")

        while index < len(tokenized_sentence):
            if index + 1 < len(tokenized_sentence) and (tokenized_sentence[index], tokenized_sentence[index + 1]) in bigram:
                sent_with_bigrams.append(
                    (tokenized_sentence[index] + "_" + tokenized_sentence[index + 1]))
                index += 1
            else:
                sent_with_bigrams.append(tokenized_sentence[index])
            index += 1
        return sent_with_bigrams

    def preprocess_text(self, text):
        text = self.expand_short_form(text)
        text = self.remove_ascii_and_numbers(text)
        text = self.remove_punc_and_special_chars(text)
        tokens = self.tokenize(text)
        tokens = [self.normalize_char_level_mismatch(token) for token in tokens]
        tokens = self.normalize_multi_words(tokens, text)
        return tokens
