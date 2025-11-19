import Link from "next/link";

export default function Home() {
  return (
    <div className="min-h-screen relative overflow-hidden bg-gradient-to-b from-[#1a0a2e] via-[#16213e] to-[#0f3460]">
      {/* Scanline overlay */}
      <div className="scanlines pointer-events-none absolute inset-0 z-10"></div>

      {/* CRT vignette effect */}
      <div className="crt-vignette pointer-events-none absolute inset-0 z-10"></div>

      {/* Main content */}
      <main className="relative z-20 flex min-h-screen flex-col items-center justify-center px-4 py-12 sm:px-6 lg:px-8">
        {/* Arcade Cabinet Container */}
        <div className="w-full max-w-6xl">
          {/* Title Section */}
          <div className="mb-12 text-center">
            <h1 className="arcade-title mb-4 text-4xl sm:text-5xl md:text-6xl lg:text-7xl font-bold text-[#39ff14] neon-green">
              OPTION PRICING
              <br />
              ARCADE
            </h1>
            <p className="arcade-subtitle text-lg sm:text-xl md:text-2xl text-[#00ffff] neon-cyan">
              Neural Network Powered Derivatives Calculator
            </p>
          </div>

          {/* Game Cards */}
          <div className="grid grid-cols-1 gap-8 mb-16 lg:grid-cols-2">
            {/* Calculator Mode Card */}
            <Link href="/calculator" className="game-card-link">
              <div className="game-card border-[#39ff14] hover:shadow-[0_0_40px_rgba(57,255,20,0.8)] group">
                <div className="mb-6 flex justify-center">
                  <div className="flex h-20 w-20 items-center justify-center rounded-lg bg-[#39ff14]/10 text-[#39ff14]">
                    <svg
                      className="h-12 w-12"
                      fill="none"
                      stroke="currentColor"
                      viewBox="0 0 24 24"
                    >
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth={2}
                        d="M9 7h6m0 10v-3m-3 3h.01M9 17h.01M9 14h.01M12 14h.01M15 11h.01M12 11h.01M9 11h.01M7 21h10a2 2 0 002-2V5a2 2 0 00-2-2H7a2 2 0 00-2 2v14a2 2 0 002 2z"
                      />
                    </svg>
                  </div>
                </div>

                <h2 className="arcade-text mb-3 text-2xl sm:text-3xl text-[#39ff14]">
                  CALCULATOR MODE
                </h2>

                <p className="arcade-text-small mb-6 text-sm sm:text-base text-gray-300">
                  Quick pricing with parameter input
                </p>

                <ul className="arcade-text-small mb-8 space-y-2 text-left text-xs sm:text-sm text-gray-400">
                  <li className="flex items-start">
                    <span className="mr-2 text-[#39ff14]">▸</span>
                    <span>408 option payoffs available</span>
                  </li>
                  <li className="flex items-start">
                    <span className="mr-2 text-[#39ff14]">▸</span>
                    <span>4 stochastic models</span>
                  </li>
                  <li className="flex items-start">
                    <span className="mr-2 text-[#39ff14]">▸</span>
                    <span>7 pricing algorithms</span>
                  </li>
                  <li className="flex items-start">
                    <span className="mr-2 text-[#39ff14]">▸</span>
                    <span>Instant results & greeks</span>
                  </li>
                </ul>

                <div className="arcade-button bg-[#39ff14] hover:bg-[#2dd10d] text-black border-[#39ff14]">
                  ENTER CALCULATOR
                </div>
              </div>
            </Link>

            {/* Interactive Mode Card */}
            <Link href="/interactive" className="game-card-link">
              <div className="game-card border-[#ff00ff] hover:shadow-[0_0_40px_rgba(255,0,255,0.8)] group">
                <div className="mb-6 flex justify-center">
                  <div className="flex h-20 w-20 items-center justify-center rounded-lg bg-[#ff00ff]/10 text-[#ff00ff]">
                    <svg
                      className="h-12 w-12"
                      fill="none"
                      stroke="currentColor"
                      viewBox="0 0 24 24"
                    >
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth={2}
                        d="M7 12l3-3 3 3 4-4M8 21l4-4 4 4M3 4h18M4 4h16v12a1 1 0 01-1 1H5a1 1 0 01-1-1V4z"
                      />
                    </svg>
                  </div>
                </div>

                <h2 className="arcade-text mb-3 text-2xl sm:text-3xl text-[#ff00ff]">
                  INTERACTIVE MODE
                </h2>

                <p className="arcade-text-small mb-6 text-sm sm:text-base text-gray-300">
                  Watch the pricing process in real-time
                </p>

                <ul className="arcade-text-small mb-8 space-y-2 text-left text-xs sm:text-sm text-gray-400">
                  <li className="flex items-start">
                    <span className="mr-2 text-[#ff00ff]">▸</span>
                    <span>Live path visualization</span>
                  </li>
                  <li className="flex items-start">
                    <span className="mr-2 text-[#ff00ff]">▸</span>
                    <span>Exercise decision tracking</span>
                  </li>
                  <li className="flex items-start">
                    <span className="mr-2 text-[#ff00ff]">▸</span>
                    <span>Animated Monte Carlo</span>
                  </li>
                  <li className="flex items-start">
                    <span className="mr-2 text-[#ff00ff]">▸</span>
                    <span>Step-by-step explanations</span>
                  </li>
                </ul>

                <div className="arcade-button bg-[#ff00ff] hover:bg-[#dd00dd] text-black border-[#ff00ff]">
                  ENTER INTERACTIVE
                </div>
              </div>
            </Link>
          </div>

          {/* Info Section */}
          <div className="text-center">
            <p className="arcade-text-small mb-4 text-sm text-[#00ffff]">
              POWERED BY:
            </p>
            <div className="flex flex-wrap justify-center gap-3">
              <div className="tech-badge bg-[#39ff14]/10 border-[#39ff14] text-[#39ff14]">
                RLSM
              </div>
              <div className="tech-badge bg-[#ff00ff]/10 border-[#ff00ff] text-[#ff00ff]">
                RFQI
              </div>
              <div className="tech-badge bg-[#00ffff]/10 border-[#00ffff] text-[#00ffff]">
                Neural Networks
              </div>
              <div className="tech-badge bg-[#ffff00]/10 border-[#ffff00] text-[#ffff00]">
                Monte Carlo
              </div>
              <div className="tech-badge bg-[#ff6600]/10 border-[#ff6600] text-[#ff6600]">
                LSM
              </div>
              <div className="tech-badge bg-[#ff0099]/10 border-[#ff0099] text-[#ff0099]">
                FQI
              </div>
            </div>
          </div>

          {/* Footer */}
          <div className="mt-12 text-center">
            <p className="arcade-text-small text-xs text-gray-500">
              INSERT COIN TO CONTINUE
            </p>
          </div>
        </div>
      </main>
    </div>
  );
}
