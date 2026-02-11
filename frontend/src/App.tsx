/**
 * HPC Dashboard - Main Application
 * Academic-style tensor data visualization dashboard
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
  extendTheme,
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

// Academic research theme - clean and refined
const theme = extendTheme({
  config: {
    initialColorMode: 'light',
    useSystemColorMode: false,
  },
  styles: {
    global: {
      body: {
        bg: '#ffffff',
        color: '#333',
      },
    },
  },
  fonts: {
    heading: '"Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif',
    body: '"Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif',
  },
  colors: {
    brand: {
      50: '#FDF2EF',
      100: '#F9DDD5',
      500: '#E07B54',
      600: '#D16A43',
      700: '#C25A33',
    },
    academic: {
      coral: '#E07B54',
      blue: '#5B8BD0',
      green: '#6BAF6B',
      text: '#333',
      textSecondary: '#666',
      textMuted: '#888',
      border: '#e0e0e0',
      bgSubtle: '#fafafa',
    },
  },
  components: {
    Button: {
      defaultProps: {
        colorScheme: 'brand',
      },
    },
    Tabs: {
      variants: {
        line: {
          tab: {
            _selected: {
              color: 'academic.coral',
              borderColor: 'academic.coral',
            },
          },
        },
      },
    },
  },
});

function Dashboard() {
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

  // Load config on mount
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

  // Execute analysis
  const handleExecute = useCallback(async () => {
    if (classWeights.length === 0) return;

    // Clear cluster selections before running new computation
    resetClusters();
    setIsLoading(true);

    try {
      const embeddingResult = await computeEmbedding(classWeights);
      setEmbeddingData(
        embeddingResult.embedding,
        embeddingResult.labels,
        embeddingResult.scaled_data,
        embeddingResult.Ms,
        embeddingResult.Mv
      );

      if (clusters.cluster1 && clusters.cluster2) {
        const analysisResult = await analyzeClusters(
          clusters.cluster1,
          clusters.cluster2,
          embeddingResult.scaled_data,
          embeddingResult.Ms,
          embeddingResult.Mv
        );
        setAnalysisResults(analysisResult.top_features, analysisResult.contribution_matrix);

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

  // Auto-analyze when both clusters are selected
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

  return (
    <Box display="flex" h="100vh" overflow="hidden" bg="#ffffff">
      {/* Sidebar */}
      <Sidebar onExecute={handleExecute} isLoading={isLoading} />

      {/* Main content area */}
      <Box flex="1" p={5} overflow="hidden" minW={0}>
        <Grid
          templateRows="1fr 1fr"
          templateColumns="1fr 1fr"
          gap={5}
          h="100%"
          w="100%"
        >
          {/* Scatter Plot (top left) */}
          <GridItem overflow="hidden" minW={0} minH={0}>
            <ScatterPlot />
          </GridItem>

          {/* Time Series (top right) */}
          <GridItem overflow="hidden" minW={0} minH={0}>
            <TimeSeriesPlot />
          </GridItem>

          {/* Feature Ranking / Heatmap tabs (bottom left) */}
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

          {/* AI Interpretation (bottom right) */}
          <GridItem overflow="hidden" minW={0} minH={0}>
            <AIInterpretation />
          </GridItem>
        </Grid>
      </Box>
    </Box>
  );
}

function App() {
  return (
    <ChakraProvider theme={theme}>
      <Dashboard />
    </ChakraProvider>
  );
}

export default App;

