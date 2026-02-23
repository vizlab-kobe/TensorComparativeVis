/**
 * メインアプリケーションコンポーネント
 *
 * テンソルデータ比較可視化ダッシュボードのルートコンポーネント。
 * 2x2 グリッドレイアウトで以下のパネルを配置する:
 *   - 左上: 散布図（PaCMAP埋め込み結果・ブラシ選択）
 *   - 右上: 時系列プロット（選択特徴量の時系列比較）
 *   - 左下: 特徴量ランキング / 寄与度ヒートマップ（タブ切り替え）
 *   - 右下: AI解釈パネル（LLM要約・履歴・比較）
 * サイドバーで TULCA 重み調整と分析実行を操作する。
 */
import { useEffect, useCallback } from 'react';
import {
  ChakraProvider,
  Box,
  Grid,
  GridItem,
  Tabs,
  TabList,
  Tab,
  TabPanels,
  TabPanel,
  useToast,
} from '@chakra-ui/react';
import {
  Sidebar,
  ScatterPlot,
  FeatureRanking,
  SpatialVisualization,
  TimeSeriesPlot,
  AIInterpretation,
} from './components';
import { useDashboardStore } from './store/dashboardStore';
import { getConfig, computeEmbedding, analyzeClusters, interpretClusters } from './api/client';
import theme from './theme';

// ---------------------------------------------------------------------------
// ダッシュボード本体コンポーネント
// ---------------------------------------------------------------------------

function Dashboard() {
  // Zustand ストアから状態とアクションを取得
  const {
    setConfig,
    classWeights,
    initializeWeights,
    clusters,
    resetClusters,
    scaledData,
    Ms,
    Mv,
    isLoading,
    setIsLoading,
    setEmbeddingData,
    setAnalysisResults,
    setInterpretation,
    activeTab,
    setActiveTab,
  } = useDashboardStore();

  const toast = useToast();

  // ── 初回マウント時に設定を読み込む ──
  useEffect(() => {
    async function loadConfig() {
      try {
        const cfg = await getConfig();
        setConfig(cfg);
        initializeWeights(cfg.n_classes);
      } catch (error) {
        toast({
          title: 'Backend not connected',
          description: 'Start the backend server: uvicorn main:app --port 8000',
          status: 'warning',
          duration: 8000,
          isClosable: true,
        });
      }
    }
    loadConfig();
  }, [setConfig, initializeWeights, toast]);

  // ── 「Execute」ボタン押下時の分析処理 ──
  const handleExecute = useCallback(async () => {
    if (classWeights.length === 0) return;

    // 新規計算前にクラスター選択をリセット
    resetClusters();
    setIsLoading(true);

    try {
      // TULCA + PaCMAP による埋め込み計算
      const embeddingResult = await computeEmbedding(classWeights);
      setEmbeddingData(
        embeddingResult.embedding,
        embeddingResult.labels,
        embeddingResult.scaled_data,
        embeddingResult.Ms,
        embeddingResult.Mv
      );

      // クラスターが既に選択されている場合は自動で分析実行
      if (clusters.cluster1 && clusters.cluster2) {
        const analysisResult = await analyzeClusters(
          clusters.cluster1,
          clusters.cluster2,
          embeddingResult.scaled_data,
          embeddingResult.Ms,
          embeddingResult.Mv
        );
        setAnalysisResults(analysisResult.top_features, analysisResult.contribution_matrix);

        // AI解釈の生成
        const interpretationResult = await interpretClusters(
          analysisResult.top_features,
          clusters.cluster1.length,
          clusters.cluster2.length
        );
        setInterpretation(interpretationResult.sections);
      }
    } catch (error) {
      console.error('Analysis error:', error);
      toast({
        title: 'Analysis failed',
        description: 'Check backend connection',
        status: 'error',
        duration: 5000,
        isClosable: true,
      });
    } finally {
      setIsLoading(false);
    }
  }, [classWeights, clusters, resetClusters, setIsLoading, setEmbeddingData, setAnalysisResults, setInterpretation, toast]);

  // ── 両クラスター選択完了時に自動分析を実行 ──
  useEffect(() => {
    async function analyzeIfReady() {
      if (clusters.cluster1 && clusters.cluster2 && scaledData && Ms && Mv) {
        setIsLoading(true);
        try {
          const analysisResult = await analyzeClusters(
            clusters.cluster1,
            clusters.cluster2,
            scaledData,
            Ms,
            Mv
          );
          setAnalysisResults(analysisResult.top_features, analysisResult.contribution_matrix);

          const interpretationResult = await interpretClusters(
            analysisResult.top_features,
            clusters.cluster1.length,
            clusters.cluster2.length
          );
          setInterpretation(interpretationResult.sections);
        } catch (error) {
          console.error('Analysis error:', error);
        } finally {
          setIsLoading(false);
        }
      }
    }
    analyzeIfReady();
  }, [clusters.cluster1, clusters.cluster2, scaledData, Ms, Mv, setIsLoading, setAnalysisResults, setInterpretation]);

  // ── レイアウト描画 ──
  return (
    <Box display="flex" h="100vh" overflow="hidden" bg="#ffffff">
      {/* サイドバー: TULCA重み調整パネル */}
      <Sidebar onExecute={handleExecute} isLoading={isLoading} />

      {/* メインコンテンツ: 2x2 グリッド */}
      <Box flex="1" p={5} overflow="hidden" minW={0}>
        <Grid
          templateRows="1fr 1fr"
          templateColumns="1fr 1fr"
          gap={5}
          h="100%"
          w="100%"
        >
          {/* 左上: 散布図（PaCMAP埋め込み） */}
          <GridItem overflow="hidden" minW={0} minH={0}>
            <ScatterPlot />
          </GridItem>

          {/* 右上: 時系列プロット */}
          <GridItem overflow="hidden" minW={0} minH={0}>
            <TimeSeriesPlot />
          </GridItem>

          {/* 左下: 特徴量ランキング / 寄与度ヒートマップ（タブ切り替え） */}
          <GridItem overflow="hidden" minW={0} minH={0}>
            <Box
              bg="white"
              borderRadius="4px"
              h="100%"
              overflow="hidden"
              border="1px solid"
              borderColor="#e0e0e0"
            >
              <Tabs
                variant="line"
                size="sm"
                index={activeTab === 'ranking' ? 0 : 1}
                onChange={(i) => setActiveTab(i === 0 ? 'ranking' : 'heatmap')}
                h="100%"
                display="flex"
                flexDirection="column"
              >
                <TabList borderBottom="1px solid" borderColor="#e0e0e0" flexShrink={0}>
                  <Tab
                    fontSize="xs"
                    fontWeight="500"
                    py={2}
                    _selected={{ color: '#555', borderColor: '#555' }}
                  >
                    Feature Contribution Ranking
                  </Tab>
                  <Tab
                    fontSize="xs"
                    fontWeight="500"
                    py={2}
                    _selected={{ color: '#555', borderColor: '#555' }}
                  >
                    Contribution Heatmap
                  </Tab>
                </TabList>
                <TabPanels flex="1" overflow="hidden">
                  <TabPanel p={0} h="100%">
                    <FeatureRanking />
                  </TabPanel>
                  <TabPanel p={0} h="100%">
                    <SpatialVisualization />
                  </TabPanel>
                </TabPanels>
              </Tabs>
            </Box>
          </GridItem>

          {/* 右下: AI解釈パネル */}
          <GridItem overflow="hidden" minW={0} minH={0}>
            <AIInterpretation />
          </GridItem>
        </Grid>
      </Box>
    </Box>
  );
}

// ---------------------------------------------------------------------------
// ルートコンポーネント（Chakra UIプロバイダーでラップ）
// ---------------------------------------------------------------------------

function App() {
  return (
    <ChakraProvider theme={theme}>
      <Dashboard />
    </ChakraProvider>
  );
}

export default App;
