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

# Trial & Error — 2026-09-18

**Obiettivo della sessione**: capire se un classificatore a closed-set basato su feature congelate di C-RADIOv4-SO400M (nessun training, solo prototipi per classe) è in grado di distinguere in modo robusto i pezzi del Mr Potato Head, come primo blocco della pipeline di video anomaly detection.

## Metodo di base

- Estrazione feature dense da C-RADIOv4-SO400M (griglia di patch 16x16px, es. 64x64 patch su immagine 1024x1024).
- Per ogni pezzo annotato a mano (poligoni labelme), calcolo del **prototipo**: media dei vettori-feature delle patch che cadono dentro il poligono.
- Classificazione di una patch nuova: similarità coseno con ciascun prototipo, assegnata la classe più vicina (nearest-centroid).
- Nessun training vero: né su C-RADIO (congelato), né su un classificatore separato — solo medie.

## Frame usati per costruire i prototipi (in ordine di aggiunta)

1. `1_disassembled_controlled_caos_f0001.jpg` — pezzi sciolti, disposizione controllata (frame di partenza).
2. `caos_f0149.jpg` — parzialmente montato.
3. `assembly_1_f0073.jpg` — completamente montato, mani presenti nella scena ma non nel poligono.
4. `5_assembled_f0002.jpg` — completamente montato, senza mani, vicino.
5. `5_assembled_f0019.jpg` — completamente montato, senza mani, angolazione leggermente diversa.

## Esperimento 1 — generalizzazione a disposizione diversa (solo frame 1)

- Test su frame con disposizione diversa dal riferimento.
- **Risultato**: `body` (pezzo grande, 66 patch nel prototipo) riconosciuto correttamente in forma e posizione. Pezzi piccoli (occhi, naso, orecchie, braccia, cappello, baffi — ognuno con 2-10 patch di supporto) confusi tra loro.
- **Interpretazione**: il "Problema A" (risoluzione/pochi pixel) e il "Problema B" (varietà di condizioni non viste) sono due cause distinte di errore, da diagnosticare separatamente.

## Esperimento 2 — corpo non riconosciuto da montato

- Test su Mr Potato Head completamente assemblato.
- **Risultato**: `body` quasi non rilevato — il prototipo di `body` veniva solo da teste "nude" isolate, troppo diverso visivamente da un corpo con la faccia sopra.
- **Fix**: aggiunti frame 2 e 3 (parzialmente/completamente montati) ai prototipi.
- **Verifica**: ritestato → `body` riconosciuto correttamente anche da montato. Fix confermato.

## Esperimento 3 — mani confuse con `arm_1`

- Su più frame con mani (guantate o nude) in scena: le mani vengono spesso classificate come `arm_1`.
- Su frame **senza** mani (es. mucchio di pezzi sciolti): il vero pezzo-braccio di plastica viene riconosciuto correttamente come `arm_1`.
- **Interpretazione**: confusione specifica "mano umana vs pezzo-braccio di plastica" (forme entrambe allungate con "dita"), non un difetto generale del metodo.
- **Stato**: diagnosticato, non ancora risolto — rimandato a una sessione dedicata (serve una classe `hand` con esempi propri).

## Esperimento 4 — naso/baffi/bocca ancora confusi dopo aggiunta frame 4 e 5

- Ipotesi testata: la confusione è dovuta a bassa risoluzione (pochi pixel per pezzo).
- **Test**: crop + upscale della zona naso/baffi/bocca, riclassificata alla stessa risoluzione effettiva più alta.
- **Risultato**: nessun miglioramento reale — la causa NON è la risoluzione.
- **Causa reale identificata**: questi pezzi compaiono "sciolti sul tavolo" in 2 frame su 3 e "montati" in 1 solo → prototipo sbilanciato verso l'aspetto isolato, stesso identico bug già risolto per `body` ma non ancora applicato a questi pezzi.
- **Fix applicato**: aggiunti frame 4 e 5 (montati, senza mani) ai prototipi.
- **Verifica su 6 frame di test diversi**: miglioramento parziale — `mouth` compare per la prima volta; `moustache` ben localizzato in alcuni test; confusione interna naso/baffi/bocca ridotta ma non eliminata del tutto.

## Esperimento 5 — validazione su stato di assemblaggio intermedio

- Ipotesi: uno stato intermedio è una "ricombinazione" di condizioni già note (pezzo isolato + pezzo montato), quindi dovrebbe generalizzare senza bisogno di annotare ogni stato intermedio possibile.
- Test su `4_partially_assembled_f0017` (non usato per i prototipi): occhi e orecchie già attaccati riconosciuti correttamente come tali (non confusi con `body`); pezzi ancora sciolti accanto localizzati bene.
- **Risultato**: ipotesi confermata sul piano puramente visivo (non sulla logica di stato di assemblaggio, che è un livello successivo dell'architettura).

## Esperimento 6 — matrice di similarità tra prototipi

- Calcolata similarità coseno tra tutti i prototipi.
- **Risultato**: `arm_1`-`arm_2` = 0.95, `pants`-`shoe` = 0.93 — nettamente più alti di qualsiasi altra coppia (es. `background` sempre sotto 0.66). Confermano un limite strutturale, non un problema di quantità di dati.

## Esperimento 7 — tentativo k-NN su tutti gli esempi pooled (invece della media)

- Ipotesi: la media (nearest-centroid) "appiattisce" varietà reale nei dati; il k-NN (confronto con ogni singolo esempio raccolto, non con la media) potrebbe recuperarla.
- Test su frame problematico: `moustache` migliora, ma `nose`/`mouth` spariscono del tutto; nuovo problema — classi con più patch raccolte (es. `ear`) "vincono" il voto a maggioranza anche dove non dovrebbero (bias da sbilanciamento nel numero di esempi per classe).
- **Decisione**: scartato. Troppo pesante per il vincolo di quasi-real-time, e non risolve comunque `arm_1`/`arm_2` e `pants`/`shoe` (confermato dalla matrice di similarità: il problema è nei singoli esempi, non nella media).

## Esperimento 8 — perché non fondere semplicemente le classi simili

- Proposta iniziale: unire `arm_1`/`arm_2` e `pants`/`shoe` in classi uniche per evitare la confusione.
- **Obiezione (corretta)**: `pants` e `shoe` sono pezzi separati che vanno montati in slot distinti nel giocattolo reale — fonderli renderebbe il sistema cieco esattamente al tipo di errore che deve rilevare (es. scarpa montata al posto sbagliato). Stesso discorso per `arm_1`/`arm_2`: il braccio sinistro/destro non è intercambiabile (confermato: pezzo fisso, non snodabile, ma orientabile nello slot).
- **Conclusione**: la fusione è accettabile solo per pezzi davvero intercambiabili nel mondo reale — non è questo il caso.

## Esperimento 9 — descrittori di forma geometrici (a costo zero, calcolati sui poligoni annotati)

**`pants` vs `shoe`**:
- Bounding box allineato agli assi: non separa bene (sensibile alla rotazione della foto).
- Rettangolo minimo ruotato (aspect ratio): `pants` = 1.79–2.51, `shoe` = 1.02–1.58 → **separazione netta, nessuna sovrapposizione**. Soglia utilizzabile: ~1.7.
- **Risolto.**

**`arm_1` vs `arm_2`**:
- Aspect ratio (anche con rettangolo ruotato): si sovrappongono, non separa (`arm_1` piegato/pugno chiuso vs `arm_2` più dritto/mano aperta — o viceversa a seconda dell'istanza — ma l'ingombro generale è simile).
- Conteggio convexity defects (insenature del contorno rispetto all'involucro convesso, normalizzato per scala): `arm_1` = 4–5 difetti, `arm_2` = 2–3 difetti → **separazione netta**.
- **Risolto** (sui poligoni annotati — vedi limite sotto).

## Limite metodologico dichiarato

I test di oggi (tranne dove specificato) usano frame provenienti dalla stessa sessione di ripresa dei frame usati per i prototipi (stessa luce, stanza, esemplare fisico). Buono per isolare il problema di identità del pezzo, **non sufficiente** per concludere che il metodo generalizza a condizioni di ripresa diverse (luce/sfondo/fotocamera diversi). Serve un test fuori sessione prima di conclusioni definitive.

## Prossimo passo deciso

I descrittori di forma dell'Esperimento 9 sono stati validati sui **poligoni disegnati a mano** (precisi). A runtime il sistema avrà solo il contorno grezzo ricavabile dalla griglia di patch (16px) prodotta dal classificatore — molto più grossolano. Prima di integrare i descrittori in una pipeline reale a blob, verificare se quella risoluzione preserva abbastanza dettaglio del contorno da far funzionare ancora il conteggio dei convexity defects (rischio concreto: i pezzi-braccio hanno solo 5-10 patch di supporto, un contorno da così poche celle potrebbe essere troppo "a scalini" per mostrare insenature nitide).

# Trial & Error — 2026-09-19
 
**Obiettivo della sessione**: chiudere (per ora) la Fase 1 (percezione), rispondendo a due domande rimaste aperte da ieri: 1) i risultati reggono anche cambiando sessione di ripresa? 2) i descrittori geometrici trovati ieri (aspect ratio, convexity defects) funzionano anche sulla griglia grezza del classificatore, non solo sui poligoni disegnati a mano? In più, costruita una demo video per il dottorando.
 
## Esperimento 1 — test cross-sessione (prototipi da una sessione, test sull'altra)
 
- Scoperta preliminare: i frame usati ieri per i prototipi erano già un mix di due sessioni di ripresa diverse (mattina A: `1_disassembled_controlled_caos_f0001`, `5_assembled_f0002`, `5_assembled_f0019`; giorno B: `caos_f0149`, `assembly_1_f0073`) — senza saperlo, alcuni test di ieri erano già in parte cross-sessione.
- Test pulito: prototipi solo da sessione A → testati su frame indipendenti di sessione B, e viceversa.
- **Risultato**: nessun crollo di prestazioni legato al cambio di sessione. I pezzi solidi ieri (hat, ear, body) restano riconoscibili attraversando la sessione; i pezzi già fragili ieri (naso/baffi/bocca, pants/shoe) restano fragili allo stesso modo, non peggiorano per il cambio di sessione.
- Unico effetto notato: prototipi di sessione B (solo 2 frame, corpo sempre già montato in entrambi) non riconoscono bene il corpo isolato/sciolto — stessa causa già nota (copertura degli stati), non un problema di sessione.
- **Conclusione**: generalizzazione cross-sessione confermata. Rafforza la validità di C-RADIOv4 come base per la percezione.
## Esperimento 2 — descrittori geometrici sulla griglia grezza (non più sui poligoni a mano)
 
- `pants` vs `shoe` (aspect ratio, rettangolo minimo ruotato): calcolato su blob rilevati dal classificatore reale (non poligoni). Valori restano nello stesso ordine di ieri (shoe più basso, pants più alto) nella maggior parte dei blob; un caso ambiguo trovato (blob etichettato `pants` con aspect ratio da `shoe`) e uno di rumore (blob troppo piccolo). **Il segnale regge, ma è più rumoroso — promettente, da calibrare meglio.**
- `arm_1` vs `arm_2` (convexity defects): sulla griglia grezza i valori collassano quasi tutti verso il basso (1-2 difetti), molto lontani dalla separazione netta di ieri (4-5 vs 2-3). **Il descrittore NON si trasferisce in modo affidabile alla risoluzione reale** — la griglia a 16px è troppo grossolana per preservare le insenature tra le dita nella maggior parte dei casi.
- **Conclusione**: rischio anticipato confermato per le braccia. `pants`/`shoe` promettente ma non ancora pronto per l'uso diretto; `arm_1`/`arm_2` resta un limite aperto, da riprendere con un approccio diverso (es. posizione relativa al corpo, non più aspetto/forma del pezzo).
## Esperimento 3 — costruzione della demo video: primo tentativo (nitidezza/blur) fallito
 
- Ipotesi iniziale: usare la nitidezza (varianza del Laplaciano) come proxy economico di "scena ferma", per invocare C-RADIO solo nei momenti di stasi (pattern compatibile col vincolo di quasi-real-time).
- Verificato sul video reale (`assembly_2.mp4`): la nitidezza varia tra 109 e 383 per tutto il video, **mai sotto la soglia di 60** impostata — il movimento in questo materiale non produce sfocatura significativa (buona illuminazione/scatto rapido). Il sistema si è bloccato in "locked" dopo pochi secondi e non ne è più uscito, ignorando i movimenti successivi.
- **Causa**: la nitidezza non è un buon proxy di movimento per questo tipo di ripresa — non un errore di soglia, un segnale sbagliato.
## Esperimento 4 — secondo tentativo: frame differencing centrale, verificato prima di ricodificare
 
- Cambiato segnale: differenza tra frame consecutivi, limitata al 50% centrale dell'immagine (per ridurre l'effetto del movimento della testa/camera, assumendo che mani/oggetto restino vicino al centro).
- Verificato sui dati reali prima di integrarlo: valori scendono sotto 1 nei momenti di stasi noti (da storyboard), salgono a 3-7 nei momenti di movimento attivo — discrimina nettamente meglio della nitidezza.
- **Integrato nella pipeline**: funziona, il video alterna correttamente scanning/locked in corrispondenza dei reali momenti di movimento e pausa.
## Esperimento 5 — bug nell'etichettatura: nomi sbagliati nonostante colori giusti
 
- Prima versione dell'overlay: raggruppava in un unico blob tutte le patch "non sfondo" adiacenti, poi assegnava il nome per voto di maggioranza su tutto il blob.
- **Bug**: quando più pezzi reali diversi sono vicini tra loro (es. occhi, naso, un frammento arm_1), finiscono in un unico mega-blob misto — il nome vincente per voto di maggioranza può cadere, come posizione del testo, sopra un pezzo completamente diverso da quello nominato.
- **Fix**: raggruppare i blob classe per classe (non "tutto ciò che non è sfondo" insieme), così ogni blob contiene una sola classe e il nome coincide sempre col colore sottostante.
## Esperimento 6 — mani dell'operatore confuse con `arm_1`/`arm_2` nel video
 
- Causa nota da ieri: mai insegnata una classe dedicata per le mani, quindi il classificatore le assegnava al pezzo più simile per forma ("dita" allungate) — il pezzo-braccio.
- Primo tentativo di fix, senza nuova annotazione: soglia sulla dimensione del blob (una mano dovrebbe creare un blob più grande di un vero pezzo-braccio). **Fallito**: con le dita divaricate, la mano si spezza in più piccoli blob separati (uno per gruppo di dita), ciascuno piccolo quanto un vero pezzo — la soglia dimensionale non li distingue.
- **Fix reale**: annotata una classe `hand` (guanto nero + pelle nuda, un frame: `caos_f0211`), aggiunta ai prototipi (6 frame totali ora). Verificato: le mani vengono ora correttamente etichettate `hand`, il pezzo-braccio vero resta `arm_1`/`arm_2` — nessuna nuova confusione introdotta.
- Nota di design: si è discusso se bastasse etichettare le mani come `background` invece di creare una classe dedicata. Scartato: il prototipo di sfondo verrebbe "diluito" mescolando tavolo liscio e pelle/mano in un'unica media poco rappresentativa di entrambi — stesso principio già visto con `body` isolato/montato.
## Esperimento 7 — falsi positivi ("oggetti che non ci sono") e fix con soglia di fiducia
 
- Osservato nel video: occasionalmente il classificatore assegna un'etichetta (es. `pants`, `mouth`) a zone di sfondo vuoto, per puro rumore — il sistema è costretto a scegliere sempre una classe, anche quando nessuna assomiglia davvero al contenuto.
- **Fix**: soglia di margine minimo di similarità sopra `background` — se nessuna classe batte `background` con margine sufficiente, si ripiega su `background` invece di forzare una scelta debole.
## Conclusioni della sessione
 
- Generalizzazione cross-sessione: confermata, nessun crollo di prestazioni.
- Descrittori geometrici: `pants`/`shoe` promettente ma da calibrare sulla griglia reale; `arm_1`/`arm_2` non trasferibile con questo approccio, resta un limite aperto.
- Demo video completata: pattern architetturale (filtro leggero sempre attivo + C-RADIO invocato solo su stasi) validato, compatibile col vincolo di quasi-real-time — non solo illustrativo, è la stessa logica prevista per il sistema finale.
- Bug via via corretti nella demo: segnale di movimento sbagliato (nitidezza → frame differencing centrale), raggruppamento blob misto (per-tutto-il-foreground → per-singola-classe), mani confuse con braccio (soglia dimensionale fallita → classe `hand` dedicata), falsi positivi su sfondo (soglia di fiducia).
## Prossimo passo deciso
 
Passare al blocco successivo dell'architettura (state tracker), riusando l'output già disponibile della percezione (nome pezzo + posizione del centroide, già calcolata per l'etichettatura del video) — utile in particolare per risolvere a valle la lateralità di `arm_1`/`arm_2` tramite posizione relativa al corpo, invece che tramite l'aspetto del pezzo. Da verificare separatamente, quando possibile: l'ipotesi (non ancora testata) che la confusione `pants`/`shoe` sia aggravata dall'occlusione del pezzo pantaloni quando montato sotto il corpo.

# Sessioni 26–29 Settembre 2026 — Notebook pulito `phase_1_perception`

## Esperimento 8 — classe `base` al posto di `pants`/`shoe`
- `pants` e `shoe` unite in un'unica classe `base` (sempre montata). Rimossi i descrittori geometrici.
- Frame di riferimento riannotati; le vecchie label `pants`/`shoe` vengono ignorate, non messe nello sfondo.

## Esperimento 9 — support set stratificato e split per sequenza
- 7 frame di riferimento scelti per stato (montato, parzialmente montato, smontato, vista dall'alto, base in due pose, mani).
- 17 frame di test da `assembly_1`, `assembly_2`, `1_disassembled_controlled_caos`, mai usati per i prototipi.
- **Scoperta:** il vecchio notebook testava su frame vicini a quelli di riferimento (stessi video) → risultati ottimisti.

## Esperimento 10 — etichettatura delle patch
- Regola a copertura (maggioranza assoluta ≥ 0.5) invece del pixel centrale; patch miste ignorate; poligoni dipinti dal più grande al più piccolo.
- **Esito:** prototipi praticamente identici alla regola del pixel centrale (coseno ≥ 0.998). Scelta tenuta perché giustificabile.

## Esperimento 11 — media normalizzata e artefatti del modello
- Prototipo = media dei vettori normalizzati. **Esito:** uguale alla media grezza (coseno ≥ 0.9998).
- **Scoperta:** 3 patch sempre anomale (riga 0, colonne 55–57), anche su un'immagine grigia → rumore a pattern fisso di C-RADIO (tech report, Sez. 2.3). Mai classificate come pezzi.

## Esperimento 12 — valutazione qualitativa
- Nessun falso positivo sullo sfondo con margin 0.03. Montaggio seguito in tutti gli stadi.
- **Errori:** braccia confuse tra loro (prototipi a 0.95); bocca di lato → naso; occhi di lato persi; **fessura vuota della bocca → bocca**.
- La fessura vuota ha un distacco top1–top2 piccolo (0.042) rispetto a una bocca vera (0.10–0.13): candidato per un criterio di confidenza.

## Esperimento 13 — latenza
- 391 ms per frame a 1024×1024 (C-RADIO 357 ms), 2.6 fps su RTX 4060 Laptop.
- **Correzione:** la misura del 14 settembre (18 ms a 512×512) era senza sincronizzazione GPU → non affidabile.

## Prossimi passi
- Annotare i 17 frame di test.
- Studio sul support set (quali e quanti frame); validazione su `assembly_2` e `1_disassembled`, test finale su `assembly_1`.
- Studio sul margin, con criterio di confidenza tra prima e seconda classe.
- Rimisurare la latenza a 512×512 per la percezione in continuità.
