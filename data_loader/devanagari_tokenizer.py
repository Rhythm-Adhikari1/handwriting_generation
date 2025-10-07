import regex as re

# --- Character classes ---
consonants = "कखगघङचछजझञटठडढणतथदधनपफबभमयरलवशषसह"
independent_vowels = "अआइईउऊऋएऐओऔ"
halant = "्"
diacritics = "ंःँ"
pre_matras = "ि"            # pre-base matra
other_matras = "ाीोौ:"      # all other matras
special_matras = "ुूृेैॄॢॣ"

# --- Regex for consonant cluster (C + (halant + C)*) ---
cluster_pattern = re.compile(f"[{consonants}](?:{halant}[{consonants}])*")

# --- Regex to match all units ---
unit_pattern = re.compile(
    f"[{independent_vowels}]"
    f"|{cluster_pattern.pattern}"
    f"|[{pre_matras}{other_matras}{diacritics}{halant}{special_matras}]"
)


def split_syllables(text):
    raw = unit_pattern.findall(text)

    result = []
    i = 0

    while i < len(raw):
        token = raw[i]

        # consonant cluster or independent vowel
        if cluster_pattern.fullmatch(token) or token in independent_vowels:
            syll = token
            j = i + 1

            # pre-base matra 'ि' goes before cluster
            if j < len(raw) and raw[j] in pre_matras:
                result.append(raw[j])
                j += 1
            
            while j < len(raw) and raw[j] in halant + diacritics + special_matras:
                syll += raw[j]
                j += 1

            result.append(syll)
            i = j

        elif token in other_matras:
            syll_2 = token
            j = i + 1
            while j < len(raw) and raw[j] in diacritics:
                syll_2 += raw[j]
                j += 1
            result.append(syll_2)
            i = j
        else:
            result.append(token)
            i += 1

    return result


# --- Test words ---
words = [
    "किरण", "काँटा", "विद्यालय", "कृष्णलीला", "सञ्जीवनी",
    "प्रयोगशाला", "शिक्षालय", "त्रिशूल", "श्रीमती", "अध्यापन",
    "ज्ञानी", "स्वास्थ्य", "स्मारकस्तम्भ", "क्लासरूम",
    "संक्रमण", "संपर्क", "सिंहावलोकन","पुस्तक","भूमि","प्रृत्ति", "ऐतिहासिक", "दृॄढ", "कॄष्णा", "तॢण", "णकज्"
]

for w in words:
    sylls = split_syllables(w)
    print(f"{w} -> {sylls}")
