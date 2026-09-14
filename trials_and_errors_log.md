Diario di Tesi — 14 Settembre 2026
1. Sintesi delle Attività Svolte

    Risoluzione Problemi di Rete (Fase -1 completata):

        Superato il blocco di download WinError 10054 sui server di Hugging Face per il modello nvidia/C-RADIOv4-SO400M implementando uno script di download incrementale con thread singolo (max_workers=1) e ripresa automatica (resume_download=True).

        Completato con successo il download dei pesi di C-RADIOv4 e del file GGUF a 1-bit di Bonsai-27B.

        Eseguito il commit di chiusura del setup nel repository (git commit su requirements e configurazioni).

    Validazione Infrasettimanale (Fase 0.1):

        Creato e strutturato il notebook Jupyter di test (Grounding Test), aggiungendo la gerarchia corretta delle intestazioni.

        Scritto ed eseguito il codice per l'inizializzazione locale di C-RADIOv4-SO400M in VRAM con supporto CUDA.

        Eseguito con successo il forward pass su un frame egocentrico di test (assembly_2_f0003.jpg), verificando il corretto shaping dei tensori in output (Summary Token [1, 2304] e Feature Spaziali 2D [1, 1152, 32, 32]).

        Registrato un tempo di inferenza a regime eccezionale (circa 18 millisecondi dopo il warmup di PyTorch), confermando la fattibilità del vincolo di "quasi real-time".

    Analisi Esplorativa Non Supervisionata (Fase 0.1.5):

        Implementata una pipeline di riduzione dimensionale tramite PCA (Principal Component Analysis) sulle feature spaziali per visualizzare lo spazio semantico appreso dal VLM senza l'ausilio di prompt testuali.

        Analizzato l'output salvato in logs/vlm_outputs/: il modello ha dimostrato un'ottima comprensione geometrica macroscopica (separazione netta tra parete verticale e piano orizzontale del tavolo) e un clustering luminoso localizzato nell'area di concentrazione dei pezzi di Mr. Potato.

        Discusso e compreso il limite strutturale della PCA sui piccoli oggetti, motivando la necessità di passare al Language-Guided Grounding tramite Dizionario Visivo.

2. Obiettivi Raggiunti

    Ambiente software, librerie GPU e pesi locali completamente blindati e funzionanti.

    Validazione tecnica superata: la pipeline di caricamento e inferenza del VLM risponde in modo stabile e ultra-rapido.

    Comprensione pratica dei limiti e delle potenzialità delle feature dense di C-RADIOv4.

3. Prossimi Obiettivi (Fase 0.2)

    Creazione del file di configurazione configs/visual_dictionary.json.

    Definizione dei primi prompt testuali strutturati (Colore + Materiale + Forma) per i componenti chiave di Mr. Potato (partendo dalle scarpe e dal corpo).

    Implementazione dello script di cosine similarity per estrarre i Bounding Box mirati a partire dalla griglia 32x32 del VLM.