import { Activity, X } from 'lucide-react';

export default function AppLayout({pageKey,sidebar,header,children,toast,toastNote='Outputs are simulated for research demonstration.',onDismiss}) {
  return <div className="app-shell">{sidebar}<div className="app-main">{header}<main key={pageKey} className="main-content">{children}</main></div>{toast&&<div className="toast" role="status"><span className="toast-icon"><Activity size={15}/></span><div><b>{toast}</b><small>{toastNote}</small></div><button aria-label="Dismiss notification" onClick={onDismiss}><X size={15}/></button></div>}</div>;
}
