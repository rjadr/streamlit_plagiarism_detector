# Streamlit Intertextual Plagiarism Detector

Streamlit app that runs a many-to-many plagiarism check on multiple documents to identify intertextual plagiarism or text reuse.

Upload multiple documents to retrieve the intertextual plagiarism scores and to explore plagiarism in the documents. File names will be treated as document IDs. GDPR notice: no documents will be stored on the server, they will be processed in memory only.

This app utilizes M.A. Sánchez Pérez et al's algorithm for text alignment and plagiarism detection.[^1][^2]

[^1]: Sanchez-Perez, M.A., Gelbukh, A., Sidorov, G.: Adaptive algorithm for plagiarism detection: The best-performing approach at PAN 2014 text alignment competition. Lecture Notes in Computer Science, vol. 9283, Springer, 2015, pp. 402-413, https://doi.org/10.1007/978-3-319-24027-5_42.

[^2]: Sanchez-Perez, M.A., Gelbukh, A.F., Sidorov, G. Dynamically adjustable approach through obfuscation type recognition. Working Notes of CLEF 2015 - Conference and Labs of the Evaluation forum, Toulouse, France, September 8-11, 2015. CEUR Workshop Proceedings, vol. 1391, CEUR-WS.org, 2015, http://ceur-ws.org/Vol-1391/92-CR.pdf. Code: https://github.com/CubasMike/plagiarism_detection_pan2015/.
