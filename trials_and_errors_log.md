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

# Registro dei Tentativi ed Errori (Trial & Error Log)

**Data:** 15 Settembre 2026  
**Progetto:** XR Assembly Anomaly Detection / Language-Guided Semantic Grounding (`thesis-yolo-pipeline` / `vad-project`)

## Sperimentazione e Risultati della Giornata

1. **Selezione interattiva dei frame:**
   - *Implementazione:* Integrazione di una finestra di dialogo (`tkinter`) in Fase 0.1 per consentire la selezione manuale dei frame direttamente dal notebook tramite interfaccia grafica, eliminando i percorsi hardcoded.

2. **Zero-Shot Text Grounding (Tentativo 1 - Cosine Similarity Diretta):**
   - *Approccio:* Calcolo della similarità a prodotto scalare tra i vettori di testo di SigLIP (dal dizionario visivo) e la griglia di feature spaziali dense di C-RADIOv4 ($32 \times 32$), seguita da normalizzazione min-max e upsampling bicubico.
   - *Esito:* **Fallito.** Le mappe di attivazione hanno generato un forte effetto "miraggio" di rumore sullo sfondo (bordi del tavolo e ombre), fallendo completamente nel rilevare i componenti più piccoli (occhi, naso, braccia).

3. **Competizione Softmax e Negative Prompting (Tentativo 2):**
   - *Approccio:* Introduzione di una classe "background" e applicazione di una Softmax scalata per temperatura per sopprimere il rumore visivo.
   - *Esito:* **Fallito.** La funzione Softmax ha agito come un "buco nero", costringendo la classe di sfondo (dominante nell'immagine) ad assorbire tutte le probabilità e azzerando le attivazioni sui piccoli oggetti.

4. **Analisi Architetturale e Revisione del Tech Report:**
   - *Analisi:* La lettura approfondita del Tech Report di C-RADIOv4 (`2601.17237.pdf`) ha evidenziato la natura agglomerativa del modello, che distilla tre teacher eterogenei: **SigLIP2** (per il summary token globale e l'allineamento testuale) e **DINOv3 / SAM3** (per le rappresentazioni spaziali e geometriche dense).
   - *Conclusione:* Il text-grounding zero-shot diretto fallisce perché la griglia spaziale densa codifica struttura e geometria, non uno spazio semantico testuale-pixel diretto accessibile tramite semplice prodotto scalare.
   - *Svolta Strategica:* Abbandonare il text-grounding diretto a favore di approcci basati su **Visual Feature Matching (corrispondenza visiva)** o **Object-Centric Tracking** (ispirati a framework avanzati come O-VAD), sfruttando la straordinaria potenza geometrica del backbone di C-RADIOv4 per l'analisi dei componenti.

   # Trial & Error Log: C-RADIOv4 Feature Extraction & Grounding Tests (16 Settembre 2026)

**Obiettivo:** Testare le capacità di visual grounding e l'estrazione di dizionari di oggetti tramite C-RADIOv4 su frame di assemblaggio egocentrico (Mr. Potato) per la pipeline di Anomaly Detection in XR.

---

### 1. Test di Estrazione Feature e Griglia Spaziale (Phase 0.1)
* **Azione:** Inizializzazione del modello C-RADIOv4-SO400M da pesi locali, esecuzione del forward pass su un frame singolo e conversione dei token in una griglia spaziale 2D ($32 \times 32$).
* **Risultato:** Tecnicamente riuscito con tempi di inferenza ottimali (circa 0.019 secondi), confermando la corretta generazione del token globale di riassunto e dei tensori spaziali densi.

### 2. Test di Visualizzazione PCA e Soppressione Outlier
* **Azione:** Applicazione di una riduzione dimensionale (PCA) sui 3 componenti principali delle feature spaziali, combinata prima con un clipping globale dei percentili e successivamente con un mascheramento dei bordi (*border-excluded fit*) per eliminare i token ad alta energia tipici dei Vision Transformer.
* **Risultato:** Parziale isolamento geometrico degli oggetti (il Mr. Potato e i componenti si staccano nettamente dal tavolo), ma persistenza di artefatti cromatici periferici e incapacità della PCA di discriminare le singole classi di oggetti tramite una semplice proiezione a 3 canali.

### 3. Test di Language-Guided Semantic Grounding (Phase 0.2)
* **Azione:** Tentativo di interrogare direttamente il modello con descrizioni testuali dei singoli componenti (es. *nose*, *hat*, *mouth*) prelevate da un dizionario visivo, calcolando la similarità a coseno rispetto alla griglia delle feature.
* **Risultato (Fallimentare / Notebook scartato):** L'approccio diretto si è rivelato totalmente inefficace. Le mappe di similarità si sono saturate o collassate sullo sfondo, dimostrando che il backbone di C-RADIOv4 non è in grado di mappare autonomamente maschere discrete o di identificare singoli oggetti testuali isolati come un dizionario di parti.

---

### Conclusioni e Decisioni Architetturali
* **Prendere atto del fallimento:** I test odierni hanno dimostrato che forzare il grounding semantico nativo sul backbone di C-RADIOv4 è un vicolo cieco per il nostro scopo.
* **Accantonamento temporaneo:** Si decide di accantonare i test basati su maschere/poligoni diretti e di rivalutare la strategia complessiva nella sessione di domani, ripensando a come integrare l'estrazione delle feature dense con il livello di ragionamento superiore.