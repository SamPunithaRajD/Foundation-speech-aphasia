export const pipeline = [
  ['Audio input','AudioWaveform','Audio signal received'],['Preprocessing','SlidersHorizontal','Noise & silence review'],['Speech foundation model','BrainCircuit','Representation extraction'],['Speech embeddings','Orbit','Acoustic encoding'],['Automatic transcription','Captions','Speech-to-text'],['Clinical language analysis','ScanText','Linguistic features'],['Multimodal fusion','Network','Feature integration'],['Severity classifier','ChartNoAxesCombined','Research classification'],['Explainability','Sparkles','Local feature attribution']
];
export const models = [
  {name:'wav2vec 2.0',tag:'Self-supervised',desc:'Contextual speech representations learned from unlabeled audio.',status:'Available'},
  {name:'HuBERT',tag:'Masked prediction',desc:'Hidden-unit discovery for robust acoustic feature learning.',status:'Available'},
  {name:'Whisper Encoder',tag:'Multilingual',desc:'Encoder representations from a large-scale speech model.',status:'Research'},
  {name:'WavLM',tag:'Speech understanding',desc:'Representations designed for varied speech tasks and conditions.',status:'Available'}
];
export const metrics = [
  ['Speech fluency','84','Higher is more fluent','↑ 8%'],['Lexical richness','61','Distinct vocabulary range','Within range'],['Lexical diversity','58','Unique words / total words','↓ 4%'],['Sentence length','7.2','Mean words per utterance','—'],['Word repetition','12%','Repeated content words','↑ 3%'],['Pause duration','1.8s','Mean between-utterance pause','↑ 0.4s'],['Pause frequency','6.4/min','Pauses per minute','—'],['Articulation rate','3.1/s','Words per speaking second','↓ 0.2'],['Speech disfluency','9%','Filled pauses & repairs','↑ 2%'],['Syntactic complexity','42','Clause structure estimate','—']
];
export const historyRows = [
 {id:'FS-2409-018',date:'24 Sep 2026',duration:'02:18',severity:'Moderate',confidence:'82.4%',model:'Multi-model',status:'Complete'},
 {id:'FS-2409-017',date:'22 Sep 2026',duration:'01:46',severity:'Mild',confidence:'76.1%',model:'wav2vec 2.0',status:'Complete'},
 {id:'FS-2409-016',date:'19 Sep 2026',duration:'03:02',severity:'Severe',confidence:'68.8%',model:'HuBERT',status:'Complete'},
 {id:'FS-2409-015',date:'16 Sep 2026',duration:'01:32',severity:'Mild',confidence:'79.6%',model:'Multi-model',status:'Complete'}
];
export const transcript = [
 {time:'00:04.2',text:'Well, I went to the… to the store yesterday.',confidence:91,flag:'Repetition'},
 {time:'00:11.8',text:'I wanted to get some apples and bread for dinner.',confidence:96},
 {time:'00:18.4',text:'But it was, um, it was already closed when I got there.',confidence:88,flag:'Filled pause'},
 {time:'00:27.1',text:'So I walked back home and called my daughter instead.',confidence:94}
];
