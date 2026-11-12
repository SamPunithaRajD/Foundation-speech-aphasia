import { useEffect, useState } from 'react';

export function useTheme() {
  const [dark,setDark]=useState(()=>localStorage.getItem('fsa-theme')!=='light');
  useEffect(()=>{
    document.documentElement.dataset.theme=dark?'dark':'light';
    localStorage.setItem('fsa-theme',dark?'dark':'light');
  },[dark]);
  return {dark,setDark};
}
