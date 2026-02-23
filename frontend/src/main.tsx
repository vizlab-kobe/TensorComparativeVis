/**
 * アプリケーション エントリーポイント
 *
 * React アプリケーションのルートレンダリングを行う。
 * StrictMode を有効にし、開発時の潜在的な問題を検出する。
 */
import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import './index.css';
import App from './App';

// ルート DOM ノードに React ツリーをマウント
createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
