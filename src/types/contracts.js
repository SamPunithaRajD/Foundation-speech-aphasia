/** @typedef {'waiting'|'processing'|'completed'|'attention'|'error'} WorkflowStatus */
/** @typedef {'live'|'simulated'} DataMode  live = real backend output; simulated = SIMULATED / DEMONSTRATION DATA */
/** @typedef {{id:string,filename:string,duration:string,simulated:boolean,mode:DataMode,createdAt:string,status:string}} AnalysisResult */
/** @typedef {{text:string,time:string,confidence:number|null,flag?:string}} TranscriptSegment  confidence: uncalibrated model probability (%) or null */
/** @typedef {{label:string,value:string,computed:boolean,explanation:string,note:string}} ClinicalFeature */
/** @typedef {{available:boolean,state:string,label?:string,confidence?:number|null,probabilities?:[string,number][]|null,message?:string}} SeverityPrediction  available=false => MODEL_NOT_READY etc.; never populated with placeholder values in live mode */
/** @typedef {{available:boolean,method?:string,rows?:[string,string,'positive'|'negative',number][],message?:string}} FeatureAttribution */
/** @typedef {{id:string,title:string,createdAt:string,simulated:boolean}} ReportResult */
// Frontend contracts; views consume these view models via src/services/*, never the HTTP API directly.
// Backend (FastAPI) contracts are documented in backend/README.md.
