/**
 * スクリーンショットボタンコンポーネント
 *
 * 指定された DOM 要素のスクリーンショットを撮影し、
 * PNG ファイルとしてダウンロードする。
 * html2canvas を使用してキャンバスに描画後、DataURL 経由で保存する。
 *
 * 使用方法:
 *   const panelRef = useRef<HTMLDivElement>(null);
 *   <ScreenshotButton targetRef={panelRef} filename="scatter_plot" />
 */
import { IconButton, Tooltip } from '@chakra-ui/react';
import html2canvas from 'html2canvas';
import React from 'react';

/** ScreenshotButton のプロパティ型定義 */
interface ScreenshotButtonProps {
    /** スクリーンショット対象の DOM 要素への ref */
    targetRef: React.RefObject<HTMLDivElement | null>;
    /** ダウンロードファイル名のプレフィックス（日付が自動付加される） */
    filename?: string;
}

export function ScreenshotButton({ targetRef, filename = 'screenshot' }: ScreenshotButtonProps) {
    /**
     * スクリーンショット撮影処理
     *
     * 1. 対象要素のスクロール・オーバーフロー制限を一時解除
     * 2. html2canvas で高解像度（2x）キャプチャ
     * 3. PNG として自動ダウンロード
     * 4. 元のスタイルを復元
     */
    const handleScreenshot = async () => {
        if (!targetRef.current) return;

        const element = targetRef.current;

        // 元のスタイルを保存（復元用）
        const originalHeight = element.style.height;
        const originalMaxHeight = element.style.maxHeight;
        const originalOverflow = element.style.overflow;
        const originalPosition = element.style.position;

        try {
            // コンテンツ全体を表示するために一時的にスタイルを変更
            element.style.height = 'auto';
            element.style.maxHeight = 'none';
            element.style.overflow = 'visible';

            // DOM リフローを待機
            await new Promise(resolve => setTimeout(resolve, 150));

            // 高解像度でキャプチャ
            const canvas = await html2canvas(element, {
                backgroundColor: '#ffffff',
                scale: 2,           // 2倍解像度（Retina対応）
                logging: false,
                useCORS: true,
                allowTaint: true,
                width: element.scrollWidth,
                height: element.scrollHeight,
            });

            // ダウンロードリンクを生成して自動クリック
            const link = document.createElement('a');
            link.download = `${filename}_${new Date().toISOString().slice(0, 10)}.png`;
            link.href = canvas.toDataURL('image/png');
            link.click();
        } catch (error) {
            console.error('スクリーンショット撮影に失敗:', error);
        } finally {
            // スタイルを元に戻す
            element.style.height = originalHeight;
            element.style.maxHeight = originalMaxHeight;
            element.style.overflow = originalOverflow;
            element.style.position = originalPosition;
        }
    };

    return (
        <Tooltip label="Save as PNG" fontSize="xs">
            <IconButton
                aria-label="Screenshot"
                icon={<CameraIcon />}
                size="xs"
                variant="ghost"
                onClick={handleScreenshot}
                color="#888"
                _hover={{ color: '#555', bg: '#f0f0f0' }}
            />
        </Tooltip>
    );
}

/** カメラアイコン（SVG インラインコンポーネント） */
function CameraIcon() {
    return (
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <rect x="3" y="5" width="18" height="14" rx="2" />
            <circle cx="12" cy="12" r="3" />
            <path d="M9 5V3h6v2" />
        </svg>
    );
}
