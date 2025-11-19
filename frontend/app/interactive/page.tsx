'use client';

import { useEffect, useRef, useState, useCallback } from 'react';

// Simulation state types
type SimulationState = 'idle' | 'running' | 'paused' | 'complete';

interface PathPoint {
  time: number;
  price: number;
}

interface SimulatedPath {
  points: PathPoint[];
  exercised: boolean;
  exerciseTime?: number;
  payoff: number;
  breachedBarrier?: boolean;
}

interface SimulationStats {
  pathsGenerated: number;
  currentPrice: number;
  avgExerciseTime: number;
  elapsedTime: number;
  paths: SimulatedPath[];
}

export default function InteractivePage() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animationFrameRef = useRef<number>();
  const startTimeRef = useRef<number>(0);

  // Simulation parameters
  const [params, setParams] = useState({
    payoff: 'MaxCall',
    model: 'BlackScholes',
    strike: 100,
    barrier: 120,
    nbPaths: 100,
    nbSteps: 50,
    maturity: 1.0,
    S0: 100,
    drift: 0.05,
    volatility: 0.2,
  });

  // Simulation state
  const [state, setState] = useState<SimulationState>('idle');
  const [stats, setStats] = useState<SimulationStats>({
    pathsGenerated: 0,
    currentPrice: 0,
    avgExerciseTime: 0,
    elapsedTime: 0,
    paths: [],
  });
  const [animationSpeed, setAnimationSpeed] = useState(1.0);
  const [fastForward, setFastForward] = useState(false);
  const [currentPathIndex, setCurrentPathIndex] = useState(0);
  const [currentStepIndex, setCurrentStepIndex] = useState(0);

  // Generate a single stock price path using Geometric Brownian Motion
  const generatePath = useCallback((seed: number): SimulatedPath => {
    const { nbSteps, maturity, S0, drift, volatility, strike, barrier } = params;
    const dt = maturity / nbSteps;
    const points: PathPoint[] = [];
    let price = S0;
    let exercised = false;
    let exerciseTime: number | undefined;
    let breachedBarrier = false;

    // Simple random number generator (deterministic with seed)
    let rng = seed;
    const random = () => {
      rng = (rng * 9301 + 49297) % 233280;
      return rng / 233280;
    };

    // Box-Muller transform for normal distribution
    const randn = () => {
      const u1 = random();
      const u2 = random();
      return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    };

    points.push({ time: 0, price: S0 });

    for (let i = 1; i <= nbSteps; i++) {
      const dW = randn() * Math.sqrt(dt);
      price = price * Math.exp((drift - 0.5 * volatility * volatility) * dt + volatility * dW);
      points.push({ time: i * dt, price });

      // Check barrier breach (up-and-out)
      if (price >= barrier) {
        breachedBarrier = true;
      }

      // Simple exercise decision (exercise if in-the-money and random chance)
      if (!exercised && i > nbSteps / 2 && price > strike && random() > 0.7) {
        exercised = true;
        exerciseTime = i * dt;
      }
    }

    // Calculate payoff
    let payoff = 0;
    if (!breachedBarrier) {
      if (exercised && exerciseTime !== undefined) {
        const exercisePrice = points[Math.floor(exerciseTime / dt)].price;
        payoff = Math.max(0, exercisePrice - strike);
      } else {
        payoff = Math.max(0, price - strike);
      }
    }

    return { points, exercised, exerciseTime, payoff, breachedBarrier };
  }, [params]);

  // Draw all paths on canvas
  const drawCanvas = useCallback((pathsToDraw: number, stepsToDraw: number) => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const width = canvas.width;
    const height = canvas.height;
    const padding = 40;
    const plotWidth = width - 2 * padding;
    const plotHeight = height - 2 * padding;

    // Clear canvas
    ctx.fillStyle = '#0a0a0a';
    ctx.fillRect(0, 0, width, height);

    // Draw grid
    ctx.strokeStyle = '#1a1a1a';
    ctx.lineWidth = 1;
    for (let i = 0; i <= 10; i++) {
      const x = padding + (i / 10) * plotWidth;
      const y = padding + (i / 10) * plotHeight;
      ctx.beginPath();
      ctx.moveTo(x, padding);
      ctx.lineTo(x, height - padding);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(padding, y);
      ctx.lineTo(width - padding, y);
      ctx.stroke();
    }

    // Draw axes
    ctx.strokeStyle = '#00ffff';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(padding, height - padding);
    ctx.lineTo(width - padding, height - padding);
    ctx.lineTo(width - padding, padding);
    ctx.stroke();

    // Draw barrier line
    const barrierY = height - padding - ((params.barrier - 50) / 100) * plotHeight;
    ctx.strokeStyle = '#ff0000';
    ctx.lineWidth = 2;
    ctx.setLineDash([5, 5]);
    ctx.beginPath();
    ctx.moveTo(padding, barrierY);
    ctx.lineTo(width - padding, barrierY);
    ctx.stroke();
    ctx.setLineDash([]);

    // Draw barrier label
    ctx.fillStyle = '#ff0000';
    ctx.font = '12px monospace';
    ctx.fillText(`Barrier: ${params.barrier}`, width - padding + 5, barrierY);

    // Draw strike line
    const strikeY = height - padding - ((params.strike - 50) / 100) * plotHeight;
    ctx.strokeStyle = '#ffff00';
    ctx.lineWidth = 1;
    ctx.setLineDash([3, 3]);
    ctx.beginPath();
    ctx.moveTo(padding, strikeY);
    ctx.lineTo(width - padding, strikeY);
    ctx.stroke();
    ctx.setLineDash([]);

    // Draw labels
    ctx.fillStyle = '#00ffff';
    ctx.font = '12px monospace';
    ctx.fillText('Time', width / 2 - 20, height - 10);
    ctx.save();
    ctx.translate(15, height / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText('Stock Price', 0, 0);
    ctx.restore();

    // Draw paths
    const { paths } = stats;
    for (let pathIdx = 0; pathIdx < Math.min(pathsToDraw, paths.length); pathIdx++) {
      const path = paths[pathIdx];
      const isCurrentPath = pathIdx === pathsToDraw - 1;
      const numPoints = isCurrentPath ? stepsToDraw : path.points.length;

      // Set path color
      if (path.breachedBarrier) {
        ctx.strokeStyle = '#ff0000aa'; // Red for barrier breach
      } else if (path.exercised) {
        ctx.strokeStyle = '#00ff00aa'; // Green for exercised
      } else {
        ctx.strokeStyle = '#00ffffff66'; // Cyan for held to maturity
      }
      ctx.lineWidth = isCurrentPath ? 2 : 1;

      ctx.beginPath();
      for (let i = 0; i < numPoints; i++) {
        const point = path.points[i];
        const x = padding + (point.time / params.maturity) * plotWidth;
        const y = height - padding - ((point.price - 50) / 100) * plotHeight;

        if (i === 0) {
          ctx.moveTo(x, y);
        } else {
          ctx.lineTo(x, y);
        }
      }
      ctx.stroke();

      // Draw exercise point if exercised
      if (path.exercised && path.exerciseTime !== undefined && numPoints > params.nbSteps / 2) {
        const exerciseIdx = Math.floor((path.exerciseTime / params.maturity) * params.nbSteps);
        if (exerciseIdx < numPoints) {
          const point = path.points[exerciseIdx];
          const x = padding + (point.time / params.maturity) * plotWidth;
          const y = height - padding - ((point.price - 50) / 100) * plotHeight;

          // Draw pulsing green dot
          ctx.fillStyle = '#00ff00';
          ctx.beginPath();
          ctx.arc(x, y, 4, 0, 2 * Math.PI);
          ctx.fill();
          ctx.strokeStyle = '#00ff00';
          ctx.lineWidth = 2;
          ctx.beginPath();
          ctx.arc(x, y, 6, 0, 2 * Math.PI);
          ctx.stroke();
        }
      }

      // Draw barrier breach point if breached
      if (path.breachedBarrier && numPoints === path.points.length) {
        const breachIdx = path.points.findIndex(p => p.price >= params.barrier);
        if (breachIdx !== -1) {
          const point = path.points[breachIdx];
          const x = padding + (point.time / params.maturity) * plotWidth;
          const y = height - padding - ((point.price - 50) / 100) * plotHeight;

          // Draw red X
          ctx.strokeStyle = '#ff0000';
          ctx.lineWidth = 3;
          ctx.beginPath();
          ctx.moveTo(x - 5, y - 5);
          ctx.lineTo(x + 5, y + 5);
          ctx.moveTo(x + 5, y - 5);
          ctx.lineTo(x - 5, y + 5);
          ctx.stroke();
        }
      }
    }

    // Draw path counter
    ctx.fillStyle = '#ffff00';
    ctx.font = 'bold 14px monospace';
    ctx.fillText(`Path ${pathsToDraw}/${params.nbPaths}`, padding, padding - 10);
  }, [stats, params]);

  // Animation loop
  useEffect(() => {
    if (state !== 'running') return;

    let lastTime = performance.now();
    const animate = (currentTime: number) => {
      const deltaTime = currentTime - lastTime;

      if (fastForward) {
        // Skip animation, generate all paths immediately
        const allPaths: SimulatedPath[] = [];
        for (let i = 0; i < params.nbPaths; i++) {
          allPaths.push(generatePath(i + 1000));
        }
        const avgPayoff = allPaths.reduce((sum, p) => sum + p.payoff, 0) / allPaths.length;
        const exercisedPaths = allPaths.filter(p => p.exercised);
        const avgExTime = exercisedPaths.length > 0
          ? exercisedPaths.reduce((sum, p) => sum + (p.exerciseTime || 0), 0) / exercisedPaths.length
          : 0;

        setStats({
          paths: allPaths,
          pathsGenerated: allPaths.length,
          currentPrice: avgPayoff,
          avgExerciseTime: avgExTime,
          elapsedTime: (currentTime - startTimeRef.current) / 1000,
        });
        setCurrentPathIndex(allPaths.length);
        setCurrentStepIndex(params.nbSteps);
        setState('complete');
        return;
      }

      // Normal animation
      if (deltaTime > (16 / animationSpeed)) { // Adjust frame rate based on speed
        lastTime = currentTime;

        setStats(prevStats => {
          let { paths, pathsGenerated } = prevStats;

          // Generate new path if needed
          if (currentPathIndex >= paths.length && pathsGenerated < params.nbPaths) {
            const newPath = generatePath(pathsGenerated + 1000);
            paths = [...paths, newPath];
            pathsGenerated++;
            setCurrentPathIndex(pathsGenerated - 1);
            setCurrentStepIndex(0);
          }

          // Advance current path animation
          if (currentPathIndex < paths.length && currentStepIndex < params.nbSteps) {
            setCurrentStepIndex(prev => Math.min(prev + 1, params.nbSteps));
          } else if (currentPathIndex < paths.length - 1 || pathsGenerated < params.nbPaths) {
            setCurrentPathIndex(prev => prev + 1);
            setCurrentStepIndex(0);
          } else if (pathsGenerated >= params.nbPaths && currentStepIndex >= params.nbSteps) {
            // Simulation complete
            setState('complete');
          }

          // Calculate stats
          const completedPaths = paths.slice(0, pathsGenerated);
          const avgPayoff = completedPaths.length > 0
            ? completedPaths.reduce((sum, p) => sum + p.payoff, 0) / completedPaths.length
            : 0;
          const exercisedPaths = completedPaths.filter(p => p.exercised);
          const avgExTime = exercisedPaths.length > 0
            ? exercisedPaths.reduce((sum, p) => sum + (p.exerciseTime || 0), 0) / exercisedPaths.length
            : 0;

          return {
            paths,
            pathsGenerated,
            currentPrice: avgPayoff,
            avgExerciseTime: avgExTime,
            elapsedTime: (currentTime - startTimeRef.current) / 1000,
          };
        });

        // Draw current state
        drawCanvas(currentPathIndex + 1, currentStepIndex + 1);
      }

      animationFrameRef.current = requestAnimationFrame(animate);
    };

    animationFrameRef.current = requestAnimationFrame(animate);

    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
    };
  }, [state, currentPathIndex, currentStepIndex, animationSpeed, fastForward, params.nbPaths, params.nbSteps, generatePath, drawCanvas]);

  // Initialize canvas on mount
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    // Set canvas size
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * window.devicePixelRatio;
    canvas.height = rect.height * window.devicePixelRatio;
    const ctx = canvas.getContext('2d');
    if (ctx) {
      ctx.scale(window.devicePixelRatio, window.devicePixelRatio);
    }

    drawCanvas(0, 0);
  }, [drawCanvas]);

  const handleStart = () => {
    startTimeRef.current = performance.now();
    setStats({
      pathsGenerated: 0,
      currentPrice: 0,
      avgExerciseTime: 0,
      elapsedTime: 0,
      paths: [],
    });
    setCurrentPathIndex(0);
    setCurrentStepIndex(0);
    setState('running');
  };

  const handlePauseResume = () => {
    setState(state === 'running' ? 'paused' : 'running');
  };

  const handleReset = () => {
    setState('idle');
    setStats({
      pathsGenerated: 0,
      currentPrice: 0,
      avgExerciseTime: 0,
      elapsedTime: 0,
      paths: [],
    });
    setCurrentPathIndex(0);
    setCurrentStepIndex(0);
    setFastForward(false);
    drawCanvas(0, 0);
  };

  return (
    <div className="min-h-screen bg-black text-white font-mono">
      {/* Arcade Cabinet Frame */}
      <div className="max-w-[1600px] mx-auto p-8">
        {/* Title */}
        <div className="text-center mb-8">
          <h1 className="text-5xl font-bold text-yellow-400 mb-2" style={{
            textShadow: '0 0 10px #ffff00, 0 0 20px #ffff00, 0 0 30px #ffff00',
          }}>
            OPTION PRICING ARCADE
          </h1>
          <div className="text-cyan-400 text-xl" style={{
            textShadow: '0 0 5px #00ffff',
          }}>
            WATCH THE MONTE CARLO SIMULATION IN ACTION
          </div>
        </div>

        {/* Main Cabinet */}
        <div className="border-4 border-yellow-400 rounded-lg p-6 bg-gradient-to-b from-gray-900 to-black"
          style={{
            boxShadow: '0 0 20px #ffff00, inset 0 0 20px rgba(255,255,0,0.1)',
          }}>

          {/* Screen Area */}
          <div className="grid grid-cols-3 gap-6 mb-6">
            {/* Left Panel - Chart Area (2/3 width) */}
            <div className="col-span-2">
              <div className="border-4 border-cyan-400 rounded-lg p-4 bg-black"
                style={{
                  boxShadow: '0 0 15px #00ffff, inset 0 0 15px rgba(0,255,255,0.1)',
                }}>
                <canvas
                  ref={canvasRef}
                  className="w-full aspect-[4/3] bg-black"
                  style={{
                    imageRendering: 'crisp-edges',
                  }}
                />
              </div>
            </div>

            {/* Right Panel - Info Display (1/3 width) */}
            <div className="space-y-4">
              {/* Parameters Panel */}
              <div className="border-2 border-cyan-400 rounded-lg p-4 bg-black/80"
                style={{
                  boxShadow: '0 0 10px #00ffff',
                }}>
                <h3 className="text-cyan-400 text-lg font-bold mb-3" style={{
                  textShadow: '0 0 5px #00ffff',
                }}>
                  PARAMETERS
                </h3>
                <div className="space-y-2 text-sm">
                  <div className="flex justify-between">
                    <span className="text-gray-400">Payoff:</span>
                    <span className="text-white font-bold">{params.payoff}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-400">Model:</span>
                    <span className="text-white font-bold">{params.model}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-400">Strike:</span>
                    <span className="text-yellow-400 font-bold">${params.strike}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-400">Barrier:</span>
                    <span className="text-red-400 font-bold">${params.barrier}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-400">Initial:</span>
                    <span className="text-white font-bold">${params.S0}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-400">Volatility:</span>
                    <span className="text-white font-bold">{(params.volatility * 100).toFixed(0)}%</span>
                  </div>
                </div>
              </div>

              {/* Stats Panel */}
              <div className="border-2 border-green-400 rounded-lg p-4 bg-black/80"
                style={{
                  boxShadow: '0 0 10px #00ff00',
                }}>
                <h3 className="text-green-400 text-lg font-bold mb-3" style={{
                  textShadow: '0 0 5px #00ff00',
                }}>
                  REAL-TIME STATS
                </h3>
                <div className="space-y-2 text-sm">
                  <div className="flex justify-between">
                    <span className="text-gray-400">Paths:</span>
                    <span className="text-green-400 font-bold text-lg">
                      {stats.pathsGenerated}/{params.nbPaths}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-400">Price Est:</span>
                    <span className="text-green-400 font-bold text-lg">
                      ${stats.currentPrice.toFixed(2)}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-400">Avg Ex. Time:</span>
                    <span className="text-white font-bold">
                      {stats.avgExerciseTime.toFixed(2)}y
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-400">Elapsed:</span>
                    <span className="text-white font-bold">
                      {stats.elapsedTime.toFixed(1)}s
                    </span>
                  </div>
                </div>
              </div>

              {/* Payoff Formula Panel */}
              <div className="border-2 border-cyan-400 rounded-lg p-4 bg-black/80"
                style={{
                  boxShadow: '0 0 10px #00ffff',
                }}>
                <h3 className="text-cyan-400 text-lg font-bold mb-3" style={{
                  textShadow: '0 0 5px #00ffff',
                }}>
                  PAYOFF FORMULA
                </h3>
                <div className="text-sm text-white bg-gray-900 p-3 rounded font-mono">
                  max(S - K, 0)
                  <br />
                  <span className="text-gray-400">if S &lt; Barrier</span>
                  <br />
                  <span className="text-red-400">else 0</span>
                </div>
              </div>

              {/* Warning Panel */}
              {stats.paths.length > 0 && stats.paths[stats.paths.length - 1]?.points.some(p => p.price > params.barrier - 5) && (
                <div className="border-2 border-red-500 rounded-lg p-4 bg-red-900/20 animate-pulse"
                  style={{
                    boxShadow: '0 0 15px #ff0000',
                  }}>
                  <div className="text-red-400 font-bold text-center text-sm">
                    ⚠️ NEAR BARRIER ⚠️
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* Controls Panel */}
          <div className="border-4 border-magenta-500 rounded-lg p-6 bg-gray-900"
            style={{
              boxShadow: '0 0 15px #ff00ff, inset 0 0 15px rgba(255,0,255,0.1)',
            }}>
            <div className="flex items-center justify-between gap-4">
              {/* Main Control Buttons */}
              <div className="flex gap-4">
                {state === 'idle' && (
                  <button
                    onClick={handleStart}
                    className="px-8 py-4 bg-cyan-500 text-black font-bold text-xl rounded-lg border-4 border-cyan-300 hover:bg-cyan-400 transition-all"
                    style={{
                      boxShadow: '0 0 20px #00ffff, 0 4px 0 #0088aa',
                    }}
                  >
                    START SIMULATION
                  </button>
                )}

                {(state === 'running' || state === 'paused') && (
                  <button
                    onClick={handlePauseResume}
                    className="px-8 py-4 bg-yellow-500 text-black font-bold text-xl rounded-lg border-4 border-yellow-300 hover:bg-yellow-400 transition-all"
                    style={{
                      boxShadow: '0 0 20px #ffff00, 0 4px 0 #aa8800',
                    }}
                  >
                    {state === 'running' ? 'PAUSE' : 'RESUME'}
                  </button>
                )}

                {state !== 'idle' && (
                  <button
                    onClick={handleReset}
                    className="px-8 py-4 bg-magenta-500 text-white font-bold text-xl rounded-lg border-4 border-magenta-300 hover:bg-magenta-400 transition-all"
                    style={{
                      boxShadow: '0 0 20px #ff00ff, 0 4px 0 #aa00aa',
                    }}
                  >
                    RESET
                  </button>
                )}
              </div>

              {/* Speed Controls */}
              <div className="flex flex-col gap-2">
                <div className="flex items-center gap-3">
                  <label className="text-white text-sm">Speed:</label>
                  <input
                    type="range"
                    min="0.25"
                    max="4"
                    step="0.25"
                    value={animationSpeed}
                    onChange={(e) => setAnimationSpeed(parseFloat(e.target.value))}
                    className="w-32"
                    disabled={state !== 'running'}
                  />
                  <span className="text-cyan-400 font-bold w-12">{animationSpeed}x</span>
                </div>
                <div className="flex items-center gap-3">
                  <label className="text-white text-sm flex items-center gap-2">
                    <input
                      type="checkbox"
                      checked={fastForward}
                      onChange={(e) => setFastForward(e.target.checked)}
                      disabled={state !== 'idle' && state !== 'running'}
                      className="w-4 h-4"
                    />
                    Fast Forward
                  </label>
                </div>
              </div>
            </div>
          </div>

          {/* Completion Message */}
          {state === 'complete' && (
            <div className="mt-6 text-center">
              <div className="inline-block border-4 border-yellow-400 rounded-lg p-6 bg-gradient-to-r from-yellow-900/50 to-orange-900/50 animate-pulse"
                style={{
                  boxShadow: '0 0 30px #ffff00',
                }}>
                <div className="text-4xl font-bold text-yellow-400 mb-2" style={{
                  textShadow: '0 0 10px #ffff00, 0 0 20px #ffff00',
                }}>
                  🎮 PRICING COMPLETE! 🎮
                </div>
                <div className="text-2xl text-white mb-4">
                  Final Option Value: <span className="text-green-400 font-bold">${stats.currentPrice.toFixed(2)}</span>
                </div>
                <div className="text-lg text-gray-300">
                  Generated {stats.pathsGenerated} paths in {stats.elapsedTime.toFixed(1)}s
                </div>
                <div className="text-sm text-cyan-400 mt-2">
                  {stats.paths.filter(p => p.exercised).length} paths exercised early •{' '}
                  {stats.paths.filter(p => p.breachedBarrier).length} barrier breaches
                </div>
              </div>
            </div>
          )}

          {/* Legend */}
          <div className="mt-6 flex justify-center gap-8 text-sm">
            <div className="flex items-center gap-2">
              <div className="w-8 h-1 bg-green-400"></div>
              <span className="text-gray-300">Exercised Early</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-8 h-1 bg-cyan-400"></div>
              <span className="text-gray-300">Held to Maturity</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-8 h-1 bg-red-400"></div>
              <span className="text-gray-300">Barrier Breach</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-4 h-4 rounded-full bg-green-400"></div>
              <span className="text-gray-300">Exercise Point</span>
            </div>
            <div className="flex items-center gap-2">
              <span className="text-red-400 font-bold">✕</span>
              <span className="text-gray-300">Barrier Hit</span>
            </div>
          </div>
        </div>

        {/* Footer */}
        <div className="text-center mt-8 text-gray-500 text-sm">
          <p>Monte Carlo Simulation • Geometric Brownian Motion • American-Style Options</p>
          <p className="mt-1">Press START to watch the pricing algorithm in real-time</p>
        </div>
      </div>
    </div>
  );
}
