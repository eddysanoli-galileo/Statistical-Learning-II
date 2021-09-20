import re
import numpy as np


# ======================================
# REGULAR EXPRESIONS
# ======================================

# Tag extraction regex
# Extracts the name and value of a tag in a step file
# - Matches # literally
# - Captures all chars from 0 to unlimited times unless they are ":" or the string start ([^:]*)
# - Matches ":" literally
# - Captures all chars from 0 to unlimited times unless they are ";" or the string start ([^:]*)
# - Matches ";" literally
tag_extraction_regex = r'#([^:]*):([^;]*);'

# Note block (measure) regex
# Extracts the blocks of steps or measures
# - Finds the start of a line (^)
# - Looks for a succession of 4 to 8 characters followed by a newline (?:.{4,8}\n)
# - The previous pattern (one line of the block) is matched 4 or more times ({4,})
# - The block has to be followed by either a "," or ";" (?:,|;)'
# Versions:
# V1 : r'^((?:.{4,8}\n){4,})(?:,|;)'
# V2 : r'((?:.{4,8}\n){4,})(?:,|;)'
measure_regex = r'((?:\n[^\n/]{4,8}){4,})\n(?:,|;)'

# Equality regex 
# Extracts both parts of multiple equalities separated by commas
# V1 : r",*(.*)=(.*)"
find_equalities_regex = r'([^,=]*)=([^,=]*)'

# Note sections regex
# Equal to: ([^:]*):([^:]*):([^:]*):([^:]*):([^:]*):([^;:]*)
# It extract the subsections that make up the tag value
notes_info_regex = r'([^:]*):' * 5 + r'([^;:]*)'


# ================================
# COMMON TYPE PARSERS
# ================================

# Parsers for common types like strings, floats, ints and bools
# Leading and trailing spaces are removed. In almost all cases, if an
# empty string is returned, the output value is set to None (except in
# the case of the bool parser, where it is set to False).

string_parser = lambda x: x.strip() if x.strip() else None
float_parser  = lambda x: float(x.strip()) if x.strip() else None
int_parser    = lambda x: int(x.strip()) if x.strip() else None
bool_parser   = lambda x: True if x.strip() == "YES" else False


# ================================
# SPECIALTY PARSERS
# ================================

# Parsers for data types exclusive to a Stepfile

# ------------------------
# EQUALITY: (BEAT = VALUE)
# For values structured as a list of equalities, like 'stops' and 'bpms'.
def equality_parser(tag_name, tag_val):

    # Empty numpy array with 0 rows. 
    # The "beat" is stored on column 0 and the associated value (bpm or stop time in secs)
    # is stored on column 1.
    beat_record = np.empty((0,2))

    # Both parts of the equalities are extracted
    # Example: (0.00000=29.00000) = (beat=equal_to)
    for beat, equal_to in re.findall(find_equalities_regex, tag_val):

        # Both the BPM and beat are parsed as floats
        beat = float_parser(beat)
        equal_to = float_parser(equal_to)

        # If a value is empty, the beat-bpm pair is skipped
        # Otherwise, the pair is concatenated vertically
        if beat == None:
            print("Missing beat. Discarding beat-value pair.")
        elif equal_to == None:
            print("Missing values. Discarding beat-bpm pair.")
        else: 
            beat_record = np.vstack((beat_record, [beat, equal_to]))

    # IF: The beat record is set to None if it has no rows
    if beat_record.size == 0:
        beat_record = None
        #print(f"Found empty values for tag '{tag_name}'. Setting parsed value to None.")

    # ELSE: The beat record is sorted in ascending "beat" order
    else:
        beat_record = beat_record[np.argsort(beat_record[:,0])]

    return beat_record

# ------------------------
# NOTES BLOCKS (MEASURES):
# For the metadata and blocks of notes under the "notes" tag
def notes_parser(raw_notes, replace_letter_arrow_types=True):

    # Dict for the different properties inside the 'notes' tag
    note_info = {}

    # Titles for the different properties
    # Order according to: https://github-wiki-see.page/m/stepmania/stepmania/wiki/sm
    info_titles = ["charttype", "description/author", "difficulty", "numericalmeter", "grooveradar", "notedata"]

    # All available chart types
    chart_types = ['dance-single', 'dance-double', 'dance-couple', 'lights-cabinet']

    # All available difficulties
    difficulties = ["Beginner", "Easy", "Medium", "Hard", "Challenge", "Edit"]

    # Valid measure (block) length or valid "beats per measure".
    beats_per_measure = [4, 8, 12, 16, 24, 32, 48, 64, 96, 192]

    # The content of each section is extracted into a tuple placed inside a list.
    # We extract the tuple and convert it into a list. 
    sections = list(re.findall(notes_info_regex, raw_notes)[0])

    # We asign values to each property
    for idx, section in enumerate(sections):

        # Property 0: Chart Type
        # Property 1: Description/Author
        # Property 2: Difficulty
        # Parsed as string
        if idx < 3:
            note_info[info_titles[idx]] = string_parser(section)

            # If the parsed chart type is non-standard
            if idx == 0 and string_parser(section) not in chart_types: 
                print(f"Nonstandard chart type '{string_parser(section)}' found.")

            # If the parsed difficulty is non-standard
            if idx == 2 and string_parser(section) not in difficulties: 
                print(f"Nonstandard difficulty '{string_parser(section)}' found.")

        # Property 3: Numerical Meter
        # Parsed as int
        elif idx == 3:
            note_info[info_titles[idx]] = int_parser(section)

        # Property 4: Groove Radar
        # Parsed as list of floats
        elif idx == 4:
            
            # Spaces are removed
            section = string_parser(section)

            # Values are split using "," as a delimiter and the resulting list
            # is then casted to float.
            note_info[info_titles[idx]] = list(map(float, section.split(",")))

        # Property 5: Note Data
        # Specialty parser
        elif idx == 5:

            # Note data is separated into measures or blocks followed by a comma
            note_blocks = re.findall(measure_regex, section)

            # Each measure is split using "\n". The resulting list of lines per
            # measure, then excludes the first part of the split (always an empty string)
            note_blocks = [note_block.split("\n")[1:] for note_block in note_blocks]

            # Now each row (string similar to '0000' or '0100') is converted into a list
            for block in range(len(note_blocks)):

                # If the number of beats per measure is non-standard
                if len(note_blocks[block]) not in beats_per_measure:
                    raise ValueError(f"Measure with non-standard amount of beats-per-measure ({len(note_blocks[block])}) found. ")

                else:
                    # The characters of the string are split
                    note_blocks[block] = [list(row) for row in note_blocks[block]]

                    # Attempt to convert to numpy array
                    # If this fails, it is probably due to a "malformed beat"
                    # (A line in a block with less columns than the others.)
                    try: 
                        note_blocks[block] = np.array(note_blocks[block])
                    except Exception as e:
                        raise Exception(f"Unable to transform measure to numpy array. Reason: {e}")

                    # Each column in a block represents an arrow: Left | Down | Up | Right
                    # The value that goes into each column represents the type of arrow:

                    # 0 - No Note
                    # 1 - Normal Note
                    # 2 - Hold Head
                    # 3 - Hold/Roll Tail
                    # 4 - Roll Head
                    # M - Mine (or "bad" note)
                    # K - Automatic keysound
                    # L - Lift note
                    # F - Fake note

                    # Due to the last ones not having a numeric value, they are assigned the
                    # next numbers on the sequence (5, 6, 7 and 8) for training.

                    # Arrow types in conjunction with their new code
                    letter_arrow_types = {"M": 5, "K": 6, "L": 7, "F": 8}

                    # We search each of the "letter" arrow types inside the current note block
                    # If the arrow type is found, it is replaced by their new code.
                    for arrow_code in letter_arrow_types.keys():

                        # The replacement is done, only if the setting is enable
                        if arrow_code in note_blocks[block] and replace_letter_arrow_types:           
                            note_blocks[block][note_blocks[block] == arrow_code] = letter_arrow_types[arrow_code]

            # Note blocks or measures are stored
            note_info[info_titles[idx]] = note_blocks

    return note_info

# ======================================
# FULL STEP FILE PARSING
# ======================================

def stepfile_parser(step_data):

    # Dict to store tag names and values
    tags = {}

    # The tag name and value are extracted
    # Example: (#TITLE: Bad Ketchup;) = (#NAME: VALUE;)
    for tag_name, tag_val in re.findall(tag_extraction_regex, step_data):

        # The tag name is turned to lowercase
        tag_name = tag_name.lower()

        # Depending on the tag name, each tag value is parsed differently
        # Tags with the same common data type are listed together. 
        # Tags that have a very specific data type are parsed individually 
        string_values   = ["title", "subtitle", "artist", "titletranslit", "subtitletranslit", "artisttranslit", "genre",
                        "credit", "banner","background", "lyricspath", "cdtitle", "music", "displaybpm", "bgchanges", 
                        "bgchanges2", "fgchanges", "keysounds", "attacks", "origin", "previewvid", "jacket", "cdimage",
                        "discimage"]
        float_values    = ["version", "offset", "samplestart", "samplelength", "musiclength", "lastbeathint"]
        int_values      = ["musicbytes"]
        bool_values     = ["selectable"]
        equality_values = ["bpms", "stops"]

        # ================================
        # PARSING ACCORDING TO TAG NAME
        # ================================

        # STRING PARSING
        if tag_name in string_values:
            tag_val_parsed = string_parser(tag_val)

        # FLOAT PARSING
        elif tag_name in float_values:
            tag_val_parsed = float_parser(tag_val)
        
        # INT PARSING
        elif tag_name in int_values:
            tag_val_parsed = int_parser(tag_val)

        # BOOL PARSING
        elif tag_name in bool_values:
            tag_val_parsed = bool_parser(tag_val)

        # EQUALITY PARSING
        elif tag_name in equality_values:
            tag_val_parsed = equality_parser(tag_name, tag_val)

        # NOTES PARSING
        elif tag_name == "notes":

            # A single song can have multiple step charts. We store those charts
            # in a list. If the "notes" key doesn't exist in the dict, we create
            # it's first element.
            if tag_name not in tags.keys():
                tag_val_parsed = [notes_parser(tag_val)]
            
            # Once the key exists, we append the new charts to the list
            else:
                tags[tag_name].append(notes_parser(tag_val))

        else:
            print(f"No parser found for the tag '{tag_name}'. Using raw value.")
            tag_val_parsed = tag_val
        
        # Parsed values are stored in the tag dict
        tags[tag_name] = tag_val_parsed

    return(tags)