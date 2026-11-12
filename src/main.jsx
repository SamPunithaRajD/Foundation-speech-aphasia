import React from 'react';
import { createRoot } from 'react-dom/client';
import App from './App.jsx';
import './styles.css';

class ErrorBoundary extends React.Component {
  constructor(props) { super(props); this.state = { error: null }; }
  static getDerivedStateFromError(error) { return { error }; }
  render() { return this.state.error ? <main style={{padding:32,color:'#eee',fontFamily:'sans-serif'}}><h1>Workspace unavailable</h1><p>The prototype could not open this view.</p><button onClick={()=>window.location.reload()}>Reload workspace</button></main> : this.props.children; }
}
createRoot(document.getElementById('root')).render(<React.StrictMode><ErrorBoundary><App /></ErrorBoundary></React.StrictMode>);
