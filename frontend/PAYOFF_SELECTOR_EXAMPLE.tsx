/**
 * Example usage of PayoffSelector component
 *
 * This file demonstrates how to integrate the PayoffSelector
 * into a Next.js page or component.
 */

'use client';

import React, { useState } from 'react';
import { PayoffSelector } from '@/components/PayoffSelector';
import { RetroPanel } from '@/components/ui/RetroPanel';
import type { PayoffInfo } from '@/lib/payoffs';

export default function PayoffSelectorExample() {
  const [selectedPayoff, setSelectedPayoff] = useState<PayoffInfo | null>(null);
  const [payoffParameters, setPayoffParameters] = useState<Record<string, any>>({});

  const handlePayoffSelect = (payoff: PayoffInfo, parameters: Record<string, any>) => {
    console.log('Payoff selected:', payoff);
    console.log('Parameters:', parameters);

    setSelectedPayoff(payoff);
    setPayoffParameters(parameters);

    // Here you would typically send this data to your backend
    // Example API call:
    // fetch('/api/configure-payoff', {
    //   method: 'POST',
    //   headers: { 'Content-Type': 'application/json' },
    //   body: JSON.stringify({ payoff: payoff.name, parameters })
    // });
  };

  return (
    <div className="min-h-screen bg-gradient-to-b from-gray-900 to-black p-8">
      <div className="max-w-6xl mx-auto space-y-8">
        {/* Title */}
        <h1
          className="text-center"
          style={{
            fontFamily: "'Press Start 2P', cursive",
            fontSize: '2rem',
            color: '#00ffff',
            textShadow: '0 0 20px #00ffff',
            marginBottom: '2rem',
          }}
        >
          Payoff Configurator
        </h1>

        {/* Payoff Selector */}
        <PayoffSelector
          onPayoffSelect={handlePayoffSelect}
          defaultCategory="Basket"
        />

        {/* Display Selected Configuration */}
        {selectedPayoff && (
          <RetroPanel borderColor="green" title="Current Configuration">
            <div className="space-y-4">
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <h3
                    style={{
                      fontFamily: "'Press Start 2P', cursive",
                      fontSize: '0.8rem',
                      color: '#00ff00',
                      marginBottom: '0.5rem',
                    }}
                  >
                    Payoff Details
                  </h3>
                  <div className="space-y-2">
                    <InfoRow label="Name" value={selectedPayoff.name} />
                    <InfoRow label="Abbreviation" value={selectedPayoff.abbreviation} />
                    <InfoRow label="Category" value={selectedPayoff.category} />
                    <InfoRow label="Subcategory" value={selectedPayoff.subcategory || 'N/A'} />
                    <InfoRow label="Path Dependent" value={selectedPayoff.isPathDependent ? 'Yes' : 'No'} />
                    <InfoRow label="Multi-Asset" value={selectedPayoff.requiresMultipleAssets ? 'Yes' : 'No'} />
                  </div>
                </div>

                <div>
                  <h3
                    style={{
                      fontFamily: "'Press Start 2P', cursive",
                      fontSize: '0.8rem',
                      color: '#00ff00',
                      marginBottom: '0.5rem',
                    }}
                  >
                    Parameters
                  </h3>
                  <div className="space-y-2">
                    {Object.entries(payoffParameters).map(([key, value]) => (
                      <InfoRow
                        key={key}
                        label={key}
                        value={Array.isArray(value) ? value.join(', ') : String(value)}
                      />
                    ))}
                    {Object.keys(payoffParameters).length === 0 && (
                      <p
                        style={{
                          fontFamily: "'Press Start 2P', cursive",
                          fontSize: '0.6rem',
                          color: '#666',
                        }}
                      >
                        No parameters set
                      </p>
                    )}
                  </div>
                </div>
              </div>

              <div>
                <h3
                  style={{
                    fontFamily: "'Press Start 2P', cursive",
                    fontSize: '0.8rem',
                    color: '#00ff00',
                    marginBottom: '0.5rem',
                  }}
                >
                  Description
                </h3>
                <p
                  style={{
                    fontFamily: "'Press Start 2P', cursive",
                    fontSize: '0.65rem',
                    color: '#aaa',
                    lineHeight: '1.6',
                  }}
                >
                  {selectedPayoff.description}
                </p>
              </div>

              {/* JSON Export */}
              <div>
                <h3
                  style={{
                    fontFamily: "'Press Start 2P', cursive",
                    fontSize: '0.8rem',
                    color: '#00ff00',
                    marginBottom: '0.5rem',
                  }}
                >
                  JSON Export
                </h3>
                <pre
                  style={{
                    fontFamily: "'Courier New', monospace",
                    fontSize: '0.7rem',
                    color: '#0f0',
                    background: 'rgba(0, 0, 0, 0.5)',
                    padding: '1rem',
                    borderRadius: '4px',
                    border: '1px solid #00ff00',
                    overflow: 'auto',
                    maxHeight: '200px',
                  }}
                >
                  {JSON.stringify(
                    {
                      payoff: selectedPayoff.name,
                      abbreviation: selectedPayoff.abbreviation,
                      parameters: payoffParameters,
                    },
                    null,
                    2
                  )}
                </pre>
              </div>
            </div>
          </RetroPanel>
        )}

        {/* Usage Instructions */}
        <RetroPanel borderColor="yellow" title="Usage Instructions">
          <div
            style={{
              fontFamily: "'Press Start 2P', cursive",
              fontSize: '0.65rem',
              color: '#ffff00',
              lineHeight: '1.8',
            }}
          >
            <ol className="space-y-2 list-decimal list-inside">
              <li>Select a category (Single Asset or Basket)</li>
              <li>Choose a base payoff from the dropdown</li>
              <li>Select a barrier type (or None for standard options)</li>
              <li>Fill in the required parameters</li>
              <li>Click "Apply Payoff" to configure</li>
            </ol>
          </div>
        </RetroPanel>
      </div>
    </div>
  );
}

/**
 * Helper component for displaying info rows
 */
interface InfoRowProps {
  label: string;
  value: string;
}

const InfoRow: React.FC<InfoRowProps> = ({ label, value }) => (
  <div className="flex justify-between items-center">
    <span
      style={{
        fontFamily: "'Press Start 2P', cursive",
        fontSize: '0.6rem',
        color: '#00ff00',
      }}
    >
      {label}:
    </span>
    <span
      style={{
        fontFamily: "'Press Start 2P', cursive",
        fontSize: '0.6rem',
        color: '#fff',
        textAlign: 'right',
      }}
    >
      {value}
    </span>
  </div>
);

/**
 * Alternative: Simpler integration example
 */
export function SimplePayoffSelectorExample() {
  return (
    <div className="p-8">
      <PayoffSelector
        onPayoffSelect={(payoff, params) => {
          // Send to API
          console.log('Selected:', payoff.name, params);

          // Or update state
          // setCurrentPayoff(payoff);
          // setCurrentParams(params);
        }}
        defaultCategory="Single Asset"
      />
    </div>
  );
}

/**
 * Example: Filter payoffs by type
 */
export function FilteredPayoffExample() {
  return (
    <div className="grid grid-cols-2 gap-8 p-8">
      {/* Single Asset Options */}
      <PayoffSelector
        defaultCategory="Single Asset"
        onPayoffSelect={(payoff, params) => {
          console.log('Single asset option:', payoff.name);
        }}
      />

      {/* Basket Options */}
      <PayoffSelector
        defaultCategory="Basket"
        onPayoffSelect={(payoff, params) => {
          console.log('Basket option:', payoff.name);
        }}
      />
    </div>
  );
}
