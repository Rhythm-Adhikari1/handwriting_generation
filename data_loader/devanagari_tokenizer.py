import regex as re

# --- Character classes ---
consonants = "कखगघङचछजझञटठडढणतथदधनपफबभमयरलवशषसह"
independent_vowels = "अआइईउऊऋएऐओऔ"
halant = "्"
diacritics = "ंःँ"
pre_matras = "ि"            # pre-base matra
other_matras = "ाीोौ:"      # all other matras
special_matras = "ुूृेैॄॢॣ"

# --- Extra characters: Nepali digits + Latin + symbols ---
extra_chars = "०१२३४५६७८९_Only thewigsofrcvdampbkuq.A-210xT5\'MDL,RYHJ\"ISPWENj&BC93VGFKz();#:!7U64Q8?+*ZX/%।"

# --- Regex for consonant cluster (C + (halant + C)*) ---
cluster_pattern = re.compile(f"[{consonants}](?:{halant}[{consonants}])*")

# --- Regex to match all units ---
unit_pattern = re.compile(
    f"[{independent_vowels}]"
    f"|{cluster_pattern.pattern}"
    f"|[{pre_matras}{other_matras}{diacritics}{halant}{special_matras}]"
    f"|[{re.escape(extra_chars)}]"  # <-- escape special regex characters
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


