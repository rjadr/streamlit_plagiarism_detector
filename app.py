import itertools
import nltk
import pandas as pd
import random
import re
import spacy
from spacy import displacy
import streamlit as st
from tika import parser as p
from PAN2015 import SGSPLAG

nltk.download('punkt')
nlp = spacy.blank("en")


def merge_intervals(intervals):
    """Merges connecting and overlapping spans in a list of spans.
    Parameters
    ----------
    intervals: list
        List of tuples representing spans/intervals
    Returns
    -------
    list
        List of merged spans
    """
    temp_tuple = [list(i) for i in intervals]
    temp_tuple.sort(key=lambda interval: interval[0])
    merged = [temp_tuple[0]]

    for current in temp_tuple:
        previous = merged[-1]
        if current[0] <= previous[1] + 1:
            previous[1] = max(previous[1], current[1])
        else:
            merged.append(current)

    return merged


def sum_of_intervals(intervals):
    """Returns the sum of the total interval length of a list of intervals.
    Parameters
    ----------
    intervals: list
        List of tuples represnting intervals/spans
    Returns
    -------
    int
        Sum of the length of all intervals
    """    
    
    number = []
    maxn = 0
    minn = 0
    for i in intervals:
        if i[1] > maxn:
            maxn = i[1]
        if i[0] < minn:
            minn = i[0]
    for i in range(0, maxn - minn):
        number.append(0)
    for i in intervals:
        for j in range(i[0]-minn, i[1]-minn):
            number[j] = 1   
    return sum(number)


def get_spacy_doc(df, text_id, text):
    """Returns a SpaCy doc and adds character spans of matches.
    Parameters
    ----------
    df: DataFrame
        Pandas DataFrame representing the compared texts and the matches locations
    text_id: string
        Id of the text to be processed
    text: string
        The body of the text to be processed
    Returns
    -------
    Doc
        SpaCy Doc with added character spans of matches
    """
    
    doc = nlp(text)
    spans = []

    def add_matches(label, matches):
        for i in matches:
            spans.append(doc.char_span(i[0], i[1], label=label.rsplit('.', 1)[0]))

    df[df['Text A'] == text_id].apply(lambda row: add_matches(row['Text B'], row['Locations in A']), axis=1)
    df[df['Text B'] == text_id].apply(lambda row: add_matches(row['Text A'], row['Locations in B']), axis=1)
    doc.spans["sc"] = spans
    return doc


@st.cache
def get_matches(text_a, text_b, parameters):
    """Retrieves text matches from two documents using SGSPLAG: https://github.com/CubasMike/plagiarism_detection_pan2015.
    Parameters
    ----------
    text_a: string
        Text A to be compared
    text_b: string
        Text B to be compared
    parameters: dictionary
        Parameters for SGSPLAG
    Returns
    -------
    list
        List of spans of matches
    """    
    sgsplag_obj = SGSPLAG(text_a,text_b, parameters)
    sgsplag_obj.process()
    return sgsplag_obj.detections


def random_color():
    """Returns a random light color.
    Returns
    -------
    string
        Hex representation of a random light color
    """
    rgb = [random.randint(128,255) for i in range(3)]
    return '#' + ''.join(f'{i:02X}' for i in rgb)


st.title('Intertextual plagiarism detection')


with st.sidebar:
    st.header('Upload files')
    uploaded_files = st.file_uploader("Upload TXT, DOC, DOCX, ODF, RTF or PDF files", accept_multiple_files=True, type=['txt', 'doc', 'docx', 'pdf', 'odf', 'rtf'])
    
    with st.form("my_form"):
        st.header('Parameters')
        th1 = st.slider('Threshold corresponding to the first similarity measure (cosine).', value=0.3, min_value=0.0, max_value=1.0, step=0.01)
        th2 = st.slider('Threshold corresponding to the second similarity measure (dice).', value=0.33, min_value=0.0, max_value=1.0, step=0.01)
        th3 = st.slider('Threshold corresponding to the third similarity measure (cosine).', value=0.34, min_value=0.0, max_value=1.0, step=0.01)
        src_gap = st.slider('Maximum gap between sentences taken as adjacents.', value=4, min_value=0, max_value=10, step=1)
        src_gap_least = st.slider('Minimum value that maximum gap can take after several iterations.', value=0, min_value=0, max_value=10, step=1)
        susp_gap = src_gap
        susp_gap_least = src_gap_least
        verbatim_minlen = st.number_input('Minimum length in characters of common substring (using words) between both documents to consider to be a verbatim obfuscation case.', value=256, min_value=0)
        src_size = st.number_input('Minimum amount of sentences in a plagiarism case in the side of the document.', value=1, min_value=0)
        susp_size = src_size
        min_sentlen = st.number_input('Minimum amount of words allowed in a sentence. If less, the sentence is anexed to the next sentence.', value=3, min_value=0)
        min_plaglen = st.number_input('Minimum amount of chacarters allowed in each side of a plagiarism case.', value=150, min_value=0)
        rssent = st.radio('Small sentences', ('Annex small sentences to the next', 'Remove small sentences'))
        rssent = 0 if rssent == 'Annex small sentences to the next' else 1
        tf_idf_p = st.radio('Define if computing tf_idf or just tf', ('Use tf', 'Compute tf-idf'))
        tf_idf_p = 0 if tf_idf_p == 'Use tf' else 1
        rem_sw = st.radio('Define the treatment of stopwords', ('Do not remove stopwords', 'Remove 50 more common stopwords', 'Remove all stopwords'))
        if rem_sw == 'Do not remove stopwords':
            rem_sw = 0
        elif rem_sw == 'Remove 50 more common stopwords':
            rem_sw = 1
        else:
            rem_sw = 3
        verbatim = st.checkbox('Use the verbatim detection method', value=True)
        verbatim = 1 if verbatim else 0
        summary = st.checkbox('Use the summary detection method')
        summary = 1 if summary else 0
        src_gap_summary = st.number_input('src_gap for the summary detection method', value=24, min_value=0)
        susp_gap_summary = st.number_input('susp_gap for the summary detection method', value=24, min_value=0)
        
        parameters = st.form_submit_button("Save")
        if parameters:
            st.session_state['parameters'] = {'th1': th1, 'th2': th2, 'th3': th3, 'src_gap': src_gap, 'src_gap_least': src_gap_least, 'susp_gap': susp_gap, 'susp_gap_least': susp_gap_least, 'verbatim_minlen': verbatim_minlen, 'src_size': src_size, 'susp_size': susp_size, 'min_sentlen': min_sentlen, 'min_plaglen': min_plaglen, 'rssent': rssent, 'tf_idf_p': tf_idf_p, 'rem_sw': rem_sw, 'verbatim': verbatim, 'summary': summary, 'src_gap_summary': src_gap_summary, 'susp_gap_summary': susp_gap_summary}
            st.experimental_rerun()

            
if 'parameters' not in st.session_state:
    st.session_state['parameters'] = {'th1': 0.3, 'th2': 0.33, 'th3': 0.34, 'src_gap': 4, 'src_gap_least': 0, 'susp_gap': 4, 'susp_gap_least': 0, 'verbatim_minlen': 256, 'src_size': 1, 'susp_size': 1, 'min_sentlen': 3, 'min_plaglen': 150, 'rssent': 0, 'tf_idf_p': 1, 'rem_sw': 0, 'verbatim': 0, 'summary': 0, 'src_gap_summary': 24, 'susp_gap_summary': 24}

    
if not uploaded_files:
    st.write("App that runs a many-to-many plagiarism check on multiple documents to identify intertextual plagiarism or text reuse.") 
    st.write("Upload multiple documents to retrieve the intertextual plagiarism scores and to explore plagiarism in the documents. File names will be treated as document IDs. GDPR notice: no documents will be stored on the server, they will be processed in memory only.")
    st.write("This app utilizes M.A. Sánchez Pérez et al's [algorithm](https://www.gelbukh.com/plagiarism-detection/PAN-2015/) for text alignment and plagiarism detection.")
    st.info("Sanchez-Perez, M.A., Gelbukh, A.F., Sidorov, G. Dynamically adjustable approach through obfuscation type recognition. Working Notes of CLEF 2015 - Conference and Labs of the Evaluation forum, Toulouse, France, September 8-11, 2015. *CEUR Workshop Proceedings*, vol. 1391, CEUR-WS.org, 2015, [http://ceur-ws.org/Vol-1391/92-CR.pdf](http://ceur-ws.org/Vol-1391/92-CR.pdf). Code: [https://github.com/CubasMike/plagiarism_detection_pan2015/](https://github.com/CubasMike/plagiarism_detection_pan2015/).")
elif len(uploaded_files) < 2:
    st.warning("Can't process a single text. Upload at least one more.")
else:
    # Load texts as dicts and remove newlines 
    texts = {i.name: {'text': re.sub(r"\s+", " ", p.from_buffer(i.read())["content"].strip())} for i in uploaded_files}

    # Create all possible text combinations
    pairs = list(itertools.combinations(texts.keys(), 2))

    # Dummy dataframe to populate
    columnLabels = ['Text A', 'Text B', 'Num Matches', 'Locations in A', 'Locations in B', 'Matching Text in A', 'Matching Text in B']
    df_pairs = pd.DataFrame(columns=columnLabels)

    my_bar = st.progress(0.0)
    prevTextObjs = {}

    for index, pair in enumerate(pairs):

        filenameA, filenameB = pair[0], pair[1]
        textA, textB = texts[filenameA]['text'], texts[filenameB]['text']

        # Put this in a dictionary so we don't have to process a file twice.
        for filename in [filenameA, filenameB]:
            if filename not in prevTextObjs:
                prevTextObjs[filename] = texts[filename]['text']

        # Just more convenient naming.
        textObjA = prevTextObjs[filenameA]
        textObjB = prevTextObjs[filenameB]

        # Reset the table of previous text objects, so we don't overload memory.
        # This means we'll only remember the previous two texts.
        prevTextObjs = {filenameA: textObjA, filenameB: textObjB}

        # Do the matching.
        matches = get_matches(textObjA, textObjB, st.session_state['parameters'])
        
        if matches:
            # Write to df
            df_pairs.loc[len(df_pairs.index)] = [pair[0], pair[1], len(matches), [i[0] for i in matches], [i[1] for i in matches], [textObjA[i[0][0]:i[0][1]] for i in matches], [textObjB[i[1][0]:i[1][1]] for i in matches]]

        my_bar.progress(index/len(pairs) + 1/len(pairs))
    
    if df_pairs.empty:
        st.warning('No matches found.')
        st.stop()
    
    # Calculate and display plagiarism percentages
    all_items = list(pd.unique(df_pairs[['Text A', 'Text B']].values.ravel('K')))
    total_spans = {i:{'name': i, 'interval_length': (length := sum_of_intervals(merge_intervals(df_pairs[df_pairs['Text A'] == i]['Locations in A'].explode().to_list() + df_pairs[df_pairs['Text B'] == i]['Locations in B'].explode().to_list()))), 'score': f'{length*100/len(texts[i]["text"]):.2f}' } for i in all_items}
    df_scores = pd.DataFrame([i for i in total_spans.values()])[['name', 'score']]
    st.dataframe(df_scores.sort_values(by=['score'], ascending=False).reset_index(drop=True))

    # Display selected text
    text_id = st.selectbox('Select a document to display its results:', [''] + all_items, format_func = lambda x: 'Select a text' if x == '' else x)

    if text_id:
        doc = get_spacy_doc(df_pairs, text_id, texts[text_id]['text'])
        colors = {k: random_color() for k in set([i.label_ for i in doc.spans['sc']])}

        st.header(f"Results {text_id.rsplit('.', 1)[0]}")
    
        html = displacy.render(
            doc,
            style="span",
            options={'colors':colors},
        )
        st.write("""<div style="overflow-x: auto; border: 1px solid #e6e9ef; border-radius: 0.25rem; padding: 1rem; margin-bottom: 2.5rem">{}</div>""".format(html.replace("\n", " ")), unsafe_allow_html=True)

        st.download_button(label = "Download as html",
                    data = html,
                    file_name = f"{text_id.rsplit('.', 1)[0]}.html",
                    mime = 'text/html')
            
        dfa = df_pairs[df_pairs['Text A'] == text_id].explode(['Locations in A', 'Locations in B', 'Matching Text in A', 'Matching Text in B'])
        dfb = df_pairs[df_pairs['Text B'] == text_id].explode(['Locations in A', 'Locations in B', 'Matching Text in A', 'Matching Text in B']).rename(columns = {'Text B':'Text A', 'Text A':'Text B', 'Locations in A':'Locations in B', 'Locations in B':'Locations in A', 'Matching Text in A': 'Matching Text in B', 'Matching Text in B':'Matching Text in A'})
        df_matches = pd.concat([dfa, dfb]).drop(columns=['Num Matches'])
        
        st.header(f"Matches database {text_id.rsplit('.', 1)[0]}")
        st.dataframe(df_matches)

        st.download_button(
            label = "Download matches as csv",
            data = df_matches.to_csv(),
            file_name = f"{text_id.rsplit('.', 1)[0]}.csv",
            mime = "text/csv"
        )
