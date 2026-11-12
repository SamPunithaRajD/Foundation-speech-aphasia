import React from 'react';
import { AudioLines, Check, Info } from 'lucide-react';

export function IconButton({label,children,onClick,className=''}) {
  return <button aria-label={label} title={label} className={`icon-button ${className}`} onClick={onClick}>{children}</button>;
}
export function Button({children,onClick,variant='primary',icon:Icon,disabled=false,className=''}) {
  return <button className={`button ${variant} ${className}`} onClick={onClick} disabled={disabled}>{Icon&&<Icon size={16}/ >}{children}</button>;
}
export function Badge({children,tone='default'}) { return <span className={`badge ${tone}`}>{children}</span>; }
export function Card({children,className='',onClick,role,tabIndex,onKeyDown}) {
  return <section className={`card ${className}`} onClick={onClick} role={role} tabIndex={tabIndex} onKeyDown={onKeyDown}>{children}</section>;
}
export function Waveform({staticWave=false}) {
  const bars=Array.from({length:52},(_,i)=>12+Math.abs(Math.sin(i*2.31))*29+Math.abs(Math.cos(i*.81))*21);
  return <div className={`waveform ${staticWave?'static':''}`} aria-hidden="true">{bars.map((v,i)=><i key={i} style={{height:`${v}%`,animationDelay:`${(i%11)*-.11}s`}}/>)}</div>;
}
export function Tooltip({label,children}) { return <span className="tooltip-wrap" tabIndex={0}><button className="tooltip-trigger" aria-label={label}><Info size={12}/></button><span role="tooltip">{children}</span></span>; }
export function Progress({value,label='Progress'}) { return <div className="progress-track" role="progressbar" aria-label={label} aria-valuemin="0" aria-valuemax="100" aria-valuenow={value}><i style={{width:`${Math.max(0,Math.min(100,value))}%`}}/></div>; }
export function EmptyState({title,description}) { return <div className="empty-state"><AudioLines size={18}/><span><b>{title}</b>{description&&<small>{description}</small>}</span></div>; }
export function StatusIcon({complete=false}) { return complete?<Check size={14}/>:<span className="waiting-dot"/>; }
