# Streamlit Intertextual Plagiarism Detector

Streamlit app that runs a many-to-many plagiarism check on multiple documents to identify intertextual plagiarism or text reuse.

Upload multiple documents to retrieve the intertextual plagiarism scores and to explore plagiarism in the documents. File names will be treated as document IDs. GDPR notice: no documents will be stored on the server, they will be processed in memory only.

This app utilizes M.A. Sánchez Pérez et al's algorithm for text alignment and plagiarism detection.[^1]

Try it: [https://rjadr-streamlit-plagiarism-detector-app-eqbrkz.streamlitapp.com/](https://rjadr-streamlit-plagiarism-detector-app-eqbrkz.streamlitapp.com/)

[^1]: Sanchez-Perez, M.A., Gelbukh, A.F., Sidorov, G. Dynamically adjustable approach through obfuscation type recognition. Working Notes of CLEF 2015 - Conference and Labs of the Evaluation forum, Toulouse, France, September 8-11, 2015. CEUR Workshop Proceedings, vol. 1391, CEUR-WS.org, 2015, http://ceur-ws.org/Vol-1391/92-CR.pdf. Code: https://github.com/CubasMike/plagiarism_detection_pan2015/.
