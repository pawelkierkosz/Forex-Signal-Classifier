import React, { useEffect, useMemo, useRef, useState } from 'react'
import axios from 'axios'

const API_URL = 'http://localhost:8000'

const toTimestamp = (time, fallback) => {
  const ts = Date.parse(time)
  return Number.isNaN(ts) ? fallback : Math.floor(ts / 1000)
}

// Konfiguracja dostępnych wskaźników
const INDICATOR_OPTS = [
  { id: 'SMA5', label: 'SMA 5', color: '#ef4444' },
  { id: 'SMA20', label: 'SMA 20', color: '#f59e0b' },
  { id: 'SMA50', label: 'SMA 50', color: '#3b82f6' },
  { id: 'EMA5', label: 'EMA 5', color: '#ec4899' },
  { id: 'EMA20', label: 'EMA 20', color: '#8b5cf6' },
  { id: 'EMA50', label: 'EMA 50', color: '#6366f1' },
  { id: 'BB_up', label: 'BB Up', color: 'rgba(56, 189, 248, 0.8)', style: 2 },
  { id: 'BB_mid', label: 'BB Mid', color: 'rgba(56, 189, 248, 0.4)', style: 2 },
  { id: 'BB_low', label: 'BB Low', color: 'rgba(56, 189, 248, 0.8)', style: 2 },

  // Wskaźniki ZigZag
  { id: 'ZZ_LINE', label: 'ZigZag Linia', color: '#00ffe1' },
  { id: 'ZZ_HIGH_PTS', label: 'ZZ Highs', color: '#00ff00', isPointSeries: true },
  { id: 'ZZ_LOW_PTS', label: 'ZZ Lows', color: '#ff4444', isPointSeries: true },
]

const buildChartData = (history) =>
  history.map((candle, idx) => ({
    ...candle,
    time: toTimestamp(candle.time ?? '', idx),
    open: Number(candle.open),
    high: Number(candle.high),
    low: Number(candle.low),
    close: Number(candle.close),
  }))

function StatTile({ label, value, hint, valueColor }) {
  return (
    <div className="tile">
      <div className="tile-label">{label}</div>
      <div className="tile-value" style={{ color: valueColor || 'inherit' }}>{value}</div>
      {hint && <div className="tile-hint">{hint}</div>}
    </div>
  )
}

function App() {
  const [history, setHistory] = useState([])
  const [historyError, setHistoryError] = useState('')
  const [loadingHistory, setLoadingHistory] = useState(true)

  const [historyLimit] = useState(8000)

  const DEFAULT_SOURCE = 'ohlc_EURUSD_H1.csv'
  const [sourceName, setSourceName] = useState(DEFAULT_SOURCE)

  const [lastFetchTime, setLastFetchTime] = useState(null)
  const [now, setNow] = useState(new Date())

  // --- MODEL (MLP) ---
  const [probability, setProbability] = useState(null)
  const [avgPivotHeight, setAvgPivotHeight] = useState(null)
  const [avgPivotWidth, setAvgPivotWidth] = useState(null)
  const [signalError, setSignalError] = useState('')

  const [loadingSignal, setLoadingSignal] = useState(false)

  // --- MODEL (LSTM) ---
  const [lstmData, setLstmData] = useState(null)
  const [lstmError, setLstmError] = useState('')
  const [loadingLstm, setLoadingLstm] = useState(false)

  const [selectedCandle, setSelectedCandle] = useState(null)

  const [zigzagDepth] = useState(23)
  const [zigzagDeviation] = useState(0.00554)
  const [zigzagBackstep] = useState(4)
  const [loadingZigzag, setLoadingZigzag] = useState(false)

  const [zigzagHighs, setZigzagHighs] = useState([])
  const [zigzagLows, setZigzagLows] = useState([])
  const [zigzagCombined, setZigzagCombined] = useState([])

  const [activeInds, setActiveInds] = useState(new Set())

  const toggleIndicator = (id) => {
    const newSet = new Set(activeInds)
    if (newSet.has(id)) newSet.delete(id)
    else newSet.add(id)
    setActiveInds(newSet)
  }

  const fetchZigzag = async (depth, deviation, backstep) => {
    setLoadingZigzag(true)
    try {
      const res = await axios.get(`${API_URL}/zigzag-csv`, {
        params: { depth, deviation, backstep }
      })

      const highs = res.data.pivot_highs || []
      const lows = res.data.pivot_lows || []
      setZigzagHighs(highs)
      setZigzagLows(lows)

      const ordered = res.data.zigzag
        ? res.data.zigzag
        : [...highs, ...lows].sort((a, b) => a.index - b.index)

      setZigzagCombined(ordered)

    } catch (err) {
      console.error("Błąd ZigZag:", err)
    } finally {
      setLoadingZigzag(false)
    }
  }

  useEffect(() => {
    if (history.length === 0) return
    const delayDebounceFn = setTimeout(() => {
      fetchZigzag(zigzagDepth, zigzagDeviation, zigzagBackstep)
    }, 500)
    return () => clearTimeout(delayDebounceFn)
  }, [zigzagDepth, zigzagDeviation, zigzagBackstep, history.length])

  useEffect(() => {
    fetchHistory(historyLimit)
  }, [])

  useEffect(() => {
    if (!historyLimit) return
    fetchHistory(historyLimit)
  }, [historyLimit])

  // Automatyczne odświeżanie wykresu i zigzag co 30 sekund
  useEffect(() => {
    const autoRefreshInterval = setInterval(() => {
      if (!loadingHistory && !loadingZigzag) {
        setHistoryError('')
        setLoadingHistory(true)
        axios.get(`${API_URL}/history`, { params: { limit: historyLimit } })
          .then(res => {
            setHistory(res.data.candles)
            setLastFetchTime(new Date())
            setSourceName(DEFAULT_SOURCE)
          })
          .catch(err => {
            setHistoryError(err.response?.data?.detail || 'Nie udało się pobrać historii z CSV.')
          })
          .finally(() => {
            setLoadingHistory(false)
          })

        fetchZigzag(zigzagDepth, zigzagDeviation, zigzagBackstep)
      }
    }, 30000)

    return () => clearInterval(autoRefreshInterval)
  }, [historyLimit, loadingHistory, loadingZigzag, zigzagDepth, zigzagDeviation, zigzagBackstep])


  // Automatyczne odświeżanie sygnałów MLP i LSTM co godzinę +1 minuta
  useEffect(() => {
    const checkHourlySignalRefresh = () => {
      const now = new Date()
      const minutes = now.getMinutes()
      const seconds = now.getSeconds()
      if (minutes === 1 && seconds <= 5) {
        requestSignal()
      }
    }
    const signalInterval = setInterval(checkHourlySignalRefresh, 1000)
    return () => clearInterval(signalInterval)
  }, [])

  const initialRetrainState = { loading: false, logs: '', success: null }
  const [mlpRetrain, setMlpRetrain] = useState(initialRetrainState)
  const [lstmRetrain, setLstmRetrain] = useState(initialRetrainState)
  const [combinedRetrain, setCombinedRetrain] = useState(initialRetrainState)

  const [isChartLibLoaded, setIsChartLibLoaded] = useState(false)

  const chartContainer = useRef(null)
  const chartInstance = useRef(null)
  const candleSeriesRef = useRef(null)

  const indicatorSeriesMap = useRef({})

  const chartData = useMemo(() => buildChartData(history), [history])

  const lastCandle = history.length ? history[history.length - 1] : null
  const candleLookup = useMemo(() => {
    const map = new Map()
    chartData.forEach((point, idx) => {
      map.set(point.time, { point, raw: history[idx] })
    })
    return map
  }, [chartData, history])

  const zigzagLineData = useMemo(() => {
    return zigzagCombined.map(p => ({
      time: Math.floor(Date.parse(p.time) / 1000),
      value: p.price
    }))
  }, [zigzagCombined])

  const zigzagHighData = useMemo(() => {
    return zigzagHighs.map(p => ({
      time: Math.floor(Date.parse(p.time) / 1000),
      value: p.price
    }))
  }, [zigzagHighs])

  const zigzagLowData = useMemo(() => {
    return zigzagLows.map(p => ({
      time: Math.floor(Date.parse(p.time) / 1000),
      value: p.price
    }))
  }, [zigzagLows])

  useEffect(() => {
    const timer = setInterval(() => setNow(new Date()), 1000)
    return () => clearInterval(timer)
  }, [])

  useEffect(() => {
    if (window.LightweightCharts) {
      setIsChartLibLoaded(true)
      return
    }
    const script = document.createElement('script')
    script.src = 'https://unpkg.com/lightweight-charts@3.8.0/dist/lightweight-charts.standalone.production.js'
    script.async = true
    script.onload = () => setIsChartLibLoaded(true)
    document.body.appendChild(script)
  }, [])

  // Główna funkcja odświeżająca wszystko
  const requestSignal = async () => {
    setSignalError('')
    setLstmError('')
    setLoadingSignal(true)
    setLoadingLstm(true)

    // Reset danych
    setAvgPivotHeight(null)
    setAvgPivotWidth(null)
    setProbability(null)
    setLstmData(null)

    Promise.allSettled([
      axios.get(`${API_URL}/predict/from-file`), // MLP
      axios.get(`${API_URL}/predict/lstm`)       // LSTM
    ]).then((results) => {
      // Obsługa MLP
      const mlpRes = results[0]
      if (mlpRes.status === 'fulfilled') {
        setProbability(mlpRes.value.data.probability)
        setAvgPivotHeight(mlpRes.value.data.avg_pivot_height_pips ?? null)
        setAvgPivotWidth(mlpRes.value.data.avg_pivot_width_bars ?? null)
      } else {
        setSignalError(mlpRes.reason.response?.data?.detail || 'Błąd modelu MLP.')
      }
      // Obsługa LSTM
      const lstmRes = results[1]
      if (lstmRes.status === 'fulfilled') {
        setLstmData(lstmRes.value.data)
      } else {
        setLstmError(lstmRes.reason.response?.data?.detail || 'Błąd modelu LSTM.')
      }

      setLoadingSignal(false)
      setLoadingLstm(false)
    })
  }

  const fetchHistory = async (limit = historyLimit) => {
    setHistoryError('')
    setLoadingHistory(true)
    setLoadingSignal(true)
    setLoadingLstm(true)

    try {
      const res = await axios.get(`${API_URL}/history`, { params: { limit } })
      setHistory(res.data.candles)
      setLastFetchTime(new Date())
      setSourceName(DEFAULT_SOURCE)
      requestSignal()
    } catch (err) {
      setHistoryError(err.response?.data?.detail || 'Nie udało się pobrać historii z CSV.')
      setLoadingSignal(false)
      setLoadingLstm(false)
    } finally {
      setLoadingHistory(false)
    }
  }

  useEffect(() => {
    fetchHistory(historyLimit)
  }, [])

  useEffect(() => {
    if (!historyLimit) return
    fetchHistory(historyLimit)
  }, [historyLimit])

  const reloadCurrentSource = () => fetchHistory(historyLimit)

  const runStreamingRetrain = async (endpoint, setState, { onChunk, onSuccessSync, onErrorSync, label } = {}) => {
    setState({ loading: true, logs: '', success: null })
    try {
      const response = await fetch(`${API_URL}${endpoint}`, { method: 'POST' })
      if (!response.body) throw new Error("Brak strumienia odpowiedzi.")
      const reader = response.body.getReader()
      const decoder = new TextDecoder("utf-8")
      while (true) {
        const { value, done } = await reader.read()
        if (done) break
        const chunk = decoder.decode(value, { stream: true })
        setState(prev => ({ ...prev, logs: prev.logs + chunk }))
        if (onChunk) onChunk(chunk)
      }
      setState(prev => ({ ...prev, success: true }))
      if (onSuccessSync) onSuccessSync()
      requestSignal()
    } catch (err) {
      console.error(err)
      const errorLine = `\n[APP ERROR${label ? `: ${label}` : ''}] Błąd połączenia: ${err.message}`
      setState(prev => ({ ...prev, success: false, logs: prev.logs + errorLine }))
      if (onChunk) onChunk(errorLine)
      if (onErrorSync) onErrorSync()
    } finally {
      setState(prev => ({ ...prev, loading: false }))
    }
  }

  const handleRetrainMlp = () => runStreamingRetrain('/retrain/mlp', setMlpRetrain)
  const handleRetrainLstm = () => runStreamingRetrain('/retrain/lstm', setLstmRetrain, { label: 'LSTM' })
  const handleRetrainBoth = () => {
    setMlpRetrain({ loading: true, logs: '', success: null })
    setLstmRetrain({ loading: true, logs: '', success: null })
    runStreamingRetrain('/retrain/both', setCombinedRetrain, {
      onChunk: (chunk) => {
        setMlpRetrain(prev => ({ ...prev, logs: prev.logs + chunk }))
        setLstmRetrain(prev => ({ ...prev, logs: prev.logs + chunk }))
      },
      onSuccessSync: () => {
        setMlpRetrain(prev => ({ ...prev, success: true, loading: false }))
        setLstmRetrain(prev => ({ ...prev, success: true, loading: false }))
      },
      onErrorSync: () => {
        setMlpRetrain(prev => ({ ...prev, success: false, loading: false }))
        setLstmRetrain(prev => ({ ...prev, success: false, loading: false }))
      },
      label: 'MLP + LSTM'
    })
  }

  // --- Widok / zoom / scroll na wykresie ---
  const setLastNBarsVisible = (n) => {
    if (!chartInstance.current || chartData.length === 0) return
    const len = chartData.length
    const fromIdx = Math.max(0, len - n)
    const fromTime = chartData[fromIdx].time
    const toTime = chartData[len - 1].time
    chartInstance.current.timeScale().setVisibleRange({ from: fromTime, to: toTime })
  }
  const zoomLast50 = () => setLastNBarsVisible(50)
  const scrollToLatest = () => {
    if (!chartInstance.current) return
    chartInstance.current.timeScale().scrollToRealTime()
  }

  // --- Inicjalizacja wykresu ---
  useEffect(() => {
    if (!isChartLibLoaded || !chartContainer.current) return
    if (!window.LightweightCharts) return
    const { createChart } = window.LightweightCharts
    const chart = createChart(chartContainer.current, {
      layout: { backgroundColor: '#0d111d', textColor: '#e5e7eb' },
      grid: { vertLines: { color: 'rgba(255,255,255,0.05)' }, horzLines: { color: 'rgba(255,255,255,0.05)' } },
      timeScale: { borderColor: 'rgba(255,255,255,0.12)', rightOffset: 12, barSpacing: 6 },
      rightPriceScale: { borderColor: 'rgba(255,255,255,0.12)' },
      width: chartContainer.current.clientWidth,
      height: 420,
    })
    candleSeriesRef.current = chart.addCandlestickSeries({
      upColor: '#22c55e', downColor: '#ef4444', borderVisible: false, wickUpColor: '#22c55e', wickDownColor: '#ef4444',
    })
    chart.timeScale().fitContent()
    indicatorSeriesMap.current = {}

    INDICATOR_OPTS.forEach(opt => {
      if (opt.id === 'ZZ_LINE') {
        const zzLine = chart.addLineSeries({ color: opt.color, lineWidth: 2, visible: false })
        indicatorSeriesMap.current[opt.id] = zzLine
        return
      }
      if (opt.isPointSeries) {
        const pointSeries = chart.addLineSeries({ color: opt.color, lineWidth: 0, visible: false })
        indicatorSeriesMap.current[opt.id] = pointSeries
        return
      }
      const series = chart.addLineSeries({ color: opt.color, lineWidth: 1, lineStyle: opt.style || 0, visible: false })
      indicatorSeriesMap.current[opt.id] = series
    })

    const handleResize = () => {
      if (chartContainer.current) chart.applyOptions({ width: chartContainer.current.clientWidth })
    }
    window.addEventListener('resize', handleResize)
    chartInstance.current = chart
    return () => {
      window.removeEventListener('resize', handleResize)
      chart.remove()
      chartInstance.current = null
      candleSeriesRef.current = null
      indicatorSeriesMap.current = {}
    }
  }, [isChartLibLoaded])

  useEffect(() => {
    if (!chartInstance.current || !candleSeriesRef.current || !chartData.length) return
    candleSeriesRef.current.setData(chartData)
    INDICATOR_OPTS.forEach(opt => {
      if (['ZZ_LINE', 'ZZ_HIGH_PTS', 'ZZ_LOW_PTS'].includes(opt.id)) return
      const series = indicatorSeriesMap.current[opt.id]
      if (!series) return
      const lineData = chartData.filter(d => d[opt.id] !== undefined && d[opt.id] !== null).map(d => ({ time: d.time, value: d[opt.id] }))
      series.setData(lineData)
      series.applyOptions({ visible: activeInds.has(opt.id) })
    })
  }, [chartData, activeInds])

  useEffect(() => {
    if (!chartInstance.current) return
    const zzLine = indicatorSeriesMap.current["ZZ_LINE"]
    const zzHighPts = indicatorSeriesMap.current["ZZ_HIGH_PTS"]
    const zzLowPts = indicatorSeriesMap.current["ZZ_LOW_PTS"]
    const isLineVisible = activeInds.has("ZZ_LINE")
    const isHighPtsVisible = activeInds.has("ZZ_HIGH_PTS")
    const isLowPtsVisible = activeInds.has("ZZ_LOW_PTS")

    if (zzLine) { zzLine.setData(zigzagLineData); zzLine.applyOptions({ visible: isLineVisible }) }
    if (zzHighPts) { zzHighPts.setData(isHighPtsVisible ? zigzagHighData : []); zzHighPts.applyOptions({ visible: isHighPtsVisible }) }
    if (zzLowPts) { zzLowPts.setData(isLowPtsVisible ? zigzagLowData : []); zzLowPts.applyOptions({ visible: isLowPtsVisible }) }
  }, [zigzagLineData, zigzagHighData, zigzagLowData, activeInds])

  useEffect(() => {
    if (!chartInstance.current) return
    const handler = (param) => {
      if (!param.time) { setSelectedCandle(null); return }
      const entry = candleLookup.get(param.time)
      if (!entry) { setSelectedCandle(null); return }
      const { point, raw } = entry
      const readableTime = raw?.time || new Date(point.time * 1000).toISOString()
      setSelectedCandle({
        time: readableTime.replace('T', ' ').replace(/\+.*/, ''),
        open: point.open, high: point.high, low: point.low, close: point.close, rsi: raw.RSI, sma20: raw.SMA20
      })
    }
    const chart = chartInstance.current
    chart.subscribeClick(handler)
    return () => chart.unsubscribeClick(handler)
  }, [candleLookup])

  const recommendationClass = useMemo(() => {
    if (probability === null) return 'neutral'
    return probability >= 0.5 ? 'buy' : 'sell'
  }, [probability])

  const recommendationCopy = useMemo(() => {
    if (probability === null) return '—'
    return probability >= 0.5 ? 'BUY' : 'SELL'
  }, [probability])

  const signalProbabilityDisplay = useMemo(() => {
    if (probability === null) return '—'
    const value = recommendationCopy === 'BUY' ? probability : 1 - probability
    return `${(value * 100).toFixed(2)}%`
  }, [probability, recommendationCopy])

  const pivotHeightDisplay = useMemo(() => {
    if (avgPivotHeight === null || avgPivotHeight === undefined) return '—'
    return `${avgPivotHeight.toFixed(2)} pips`
  }, [avgPivotHeight])

  const pivotWidthDisplay = useMemo(() => {
    if (avgPivotWidth === null || avgPivotWidth === undefined) return '—'
    return `${avgPivotWidth} świec`
  }, [avgPivotWidth])

  const secondsAgo = useMemo(() => {
    if (!lastFetchTime) return 0
    const diff = Math.floor((now.getTime() - lastFetchTime.getTime()) / 1000)
    return Math.max(0, diff)
  }, [now, lastFetchTime])

  const lstmDirectionColor = useMemo(() => {
    if (!lstmData) return '#e5e7eb'
    return lstmData.direction === 'UP' ? '#22c55e' : '#ef4444'
  }, [lstmData])

  const lstmArrow = useMemo(() => {
    if (!lstmData) return ''
    return lstmData.direction === 'UP' ? '↗' : '↘'
  }, [lstmData])

  return (
    <div className="container">
      <header className="hero">
        <div><h1>Panel tradera z podglądem rynku</h1></div>
        <div className="hero-badge"><span className="pulse" />Dane łączone bezpośrednio z MT5</div>
      </header>

      <section className="card">
        <div className="card-header">
          <div><p className="eyebrow">Analiza AI</p><h2>Sygnały i prognozy</h2></div>
          <div style={{ display: 'flex', gap: '10px' }}>
            <button onClick={requestSignal} disabled={loadingSignal || !history.length}>
              {loadingSignal || loadingLstm ? 'Analizowanie...' : 'Aktualizuj sygnały'}
            </button>
          </div>
        </div>

        {signalError && <div className="error banner">Błąd MLP: {signalError}</div>}
        {lstmError && <div className="error banner">Błąd LSTM: {lstmError}</div>}

        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px' }}>
          <div className="sub-card" style={{ background: 'rgba(255,255,255,0.02)', padding: '15px', borderRadius: '8px' }}>
            <h3 style={{ margin: '0 0 15px 0', fontSize: '15px', color: '#94a3b8' }}>Klasyfikacja Trendu (MLP)</h3>
            <div className="signal-box">
              <div><p className="label">Prawdopodobieństwo sygnału</p><p className="signal-value">{signalProbabilityDisplay}</p></div>
              <div>
                <p className="label">Sygnał</p>
                {recommendationCopy === '—' ? <p className="signal-value">—</p> : <p className={`signal-pill ${recommendationClass}`}>{recommendationCopy}</p>}
              </div>
              <div><p className="label">Śr. wys. pivotów</p><p className="signal-value">{pivotHeightDisplay}</p></div>
              <div><p className="label">Śr. szer. pivotów</p><p className="signal-value">{pivotWidthDisplay}</p></div>
            </div>
          </div>
          <div className="sub-card" style={{ background: 'rgba(255,255,255,0.02)', padding: '15px', borderRadius: '8px' }}>
            <h3 style={{ margin: '0 0 15px 0', fontSize: '15px', color: '#94a3b8' }}>Prognozowana wysokość pivotu (LSTM)</h3>
            <div className="signal-box">
              <div>
                <p className="label">Prognozowana wysokość pivotu</p>
                <p className="signal-value" style={{ color: lstmData ? '#fbbf24' : 'inherit' }}>
                  {lstmData ? lstmData.predicted_next_close.toFixed(5) : '—'}
                </p>
              </div>
              <div>
                <p className="label">Kierunek</p>
                {lstmData ? <p className="signal-value" style={{ color: lstmDirectionColor, fontWeight: 'bold' }}>{lstmArrow} {lstmData.direction}</p> : <p className="signal-value">—</p>}
              </div>
              <div>
                <p className="label">Odległość do prognozowanego pivotu</p>
                <p className="signal-value">
                  {lstmData ? `${Math.abs(lstmData.predicted_movement_pips).toFixed(1)} pips` : '—'}
                </p>
              </div>
              <div>
                <p className="label">&nbsp; </p>
                <p className="signal-value"> &nbsp;</p>
              </div>
            </div>
          </div>
        </div>
        <p className="hint" style={{ marginTop: '15px' }}>
          MLP ocenia trend (ZigZag), a LSTM przewiduje wysokość pivotu.
        </p>
      </section>

      <section className="card">
        <div className="card-header">
          <div><p className="eyebrow">Zarządzanie modelem</p><h2>Douczanie</h2></div>
          <div className="retrain-actions">
            <button onClick={handleRetrainMlp} disabled={mlpRetrain.loading || combinedRetrain.loading} style={{ background: (mlpRetrain.loading || combinedRetrain.loading) ? '#4b5563' : 'linear-gradient(120deg, #ea580c, #c2410c)' }} className="secondary">{mlpRetrain.loading ? 'Trenowanie MLP...' : 'Doucz Model MLP'}</button>
            <button onClick={handleRetrainLstm} disabled={lstmRetrain.loading || combinedRetrain.loading} style={{ background: (mlpRetrain.loading || combinedRetrain.loading) ? '#4b5563' : 'linear-gradient(120deg, #ea580c, #c2410c)' }} className="secondary">{lstmRetrain.loading ? 'Trenowanie LSTM...' : 'Doucz Model LSTM'}</button>
            <button onClick={handleRetrainBoth} disabled={combinedRetrain.loading} style={{ background: (mlpRetrain.loading || combinedRetrain.loading) ? '#4b5563' : 'linear-gradient(120deg, #d32f2f, #c62828)' }} className="secondary">{combinedRetrain.loading ? 'Trenowanie obu...' : 'Doucz oba modele'}</button>
          </div>
        </div>
        <p className="hint" style={{ marginTop: '12px' }}>Uruchom douczanie konkretnego modelu lub obu naraz. Proces odbywa się w tle.</p>
        <div style={{ marginTop: '15px' }}>
          {mlpRetrain.success === true && <div className="pill" style={{ background: 'rgba(34,197,94,0.15)', color: '#bbf7d0', width: '100%', padding: '10px', textAlign: 'center', borderColor: 'rgba(34,197,94,0.4)' }}>✅ MLP: Douczanie zakończone sukcesem.</div>}
          {mlpRetrain.success === false && <div className="error banner">❌ MLP: Błąd douczania.</div>}
          {lstmRetrain.success === true && <div className="pill" style={{ background: 'rgba(34,197,94,0.15)', color: '#bbf7d0', width: '100%', marginTop: '5px', padding: '10px', textAlign: 'center', borderColor: 'rgba(34,197,94,0.4)' }}>✅ LSTM: Douczanie zakończone sukcesem.</div>}
          {lstmRetrain.success === false && <div className="error banner" style={{ marginTop: '5px' }}>❌ LSTM: Błąd douczania.</div>}
        </div>
      </section>

      <section className="card">
        <div className="card-header">
          <div><p className="eyebrow">Wizualizacja</p><h2>Wykres OHLC EURUSD</h2></div>
          <div className="pill">{loadingHistory ? 'Ładowanie...' : `${history.length} świec`}</div>
        </div>
        <div style={{ margin: '15px 0', display: 'flex', flexWrap: 'wrap', gap: '8px' }}>
          {INDICATOR_OPTS.map(opt => (
            <label key={opt.id} className="pill subtle" style={{ cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '6px', borderColor: activeInds.has(opt.id) ? opt.color : 'rgba(148, 163, 184, 0.25)', background: activeInds.has(opt.id) ? 'rgba(255,255,255,0.05)' : 'transparent' }}>
              <input type="checkbox" checked={activeInds.has(opt.id)} onChange={() => toggleIndicator(opt.id)} style={{ accentColor: opt.color }} />
              <span style={{ color: activeInds.has(opt.id) ? opt.color : '#94a3b8', fontWeight: 600, fontSize: '13px' }}>{opt.label}</span>
            </label>
          ))}
        </div>
        {historyError ? <div className="error banner">{historyError}</div> : (
          <>
            <div className="chart" ref={chartContainer} aria-label="Wykres świecowy EURUSD">
              {!isChartLibLoaded && <div style={{ padding: 20 }}>Ładowanie biblioteki wykresów...</div>}
            </div>
            <div className="grid">
              <StatTile label="Ostatnie zamknięcie" value={lastCandle ? lastCandle.close.toFixed(5) : '—'} hint={lastFetchTime ? `Aktualizacja: ${lastFetchTime.toLocaleTimeString('pl-PL')} (${secondsAgo}s temu)` : '...'} />
              <StatTile label="Wysokość ostatniej świecy" value={lastCandle ? `${((lastCandle.high - lastCandle.low) * 10000).toFixed(2)} pips` : '—'} hint={lastCandle ? `High–Low: ${lastCandle.low.toFixed(5)} → ${lastCandle.high.toFixed(5)}` : ''} />
            </div>
            <div className="controls inline" style={{ display: "flex", justifyContent: "space-between", alignItems: "center", flexWrap: "wrap", width: "100%" }}>
              <div className="pill">Zasięg: 8000 świec</div>
              <div style={{ display: "flex", gap: "12px" }}>
                <button onClick={reloadCurrentSource} disabled={loadingHistory}>{loadingHistory ? 'Ładowanie...' : 'Aktualizuj wykres'}</button>
                <button onClick={zoomLast50}>Ostatnie 50 świec</button>
                <button onClick={scrollToLatest}>Ostatnia świeca →</button>
              </div>
            </div>
            {selectedCandle && (
              <div className="info-bar" role="status" aria-live="polite">
                <div><p className="label">Czas świecy</p><p className="info-value">{selectedCandle.time}</p></div>
                <div className="info-grid">
                  <div><p className="label">Open</p><p className="info-value">{selectedCandle.open.toFixed(5)}</p></div>
                  <div><p className="label">High</p><p className="info-value">{selectedCandle.high.toFixed(5)}</p></div>
                  <div><p className="label">Low</p><p className="info-value">{selectedCandle.low.toFixed(5)}</p></div>
                  <div><p className="label">Close</p><p className="info-value">{selectedCandle.close.toFixed(5)}</p></div>
                </div>
              </div>
            )}
          </>
        )}
      </section>
    </div>
  )
}

export default App