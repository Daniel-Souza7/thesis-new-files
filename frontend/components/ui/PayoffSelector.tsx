'use client';

import React, { useState, useMemo } from 'react';
import {
  ALL_PAYOFFS,
  BARRIER_TYPES,
  BARRIER_DESCRIPTIONS,
  getPayoffsByCategory,
  getBasePayoffs,
  getBarrierParameters,
  type PayoffInfo,
  type PayoffCategory,
  type BarrierType,
  type PayoffParameter,
} from '@/lib/payoffs';
import { RetroPanel } from './RetroPanel';

interface PayoffSelectorProps {
  onPayoffSelect?: (payoff: PayoffInfo, parameters: Record<string, any>) => void;
  defaultCategory?: PayoffCategory;
}

export const PayoffSelector: React.FC<PayoffSelectorProps> = ({
  onPayoffSelect,
  defaultCategory = 'Single Asset',
}) => {
  const [selectedCategory, setSelectedCategory] = useState<PayoffCategory>(defaultCategory);
  const [selectedBasePayoff, setSelectedBasePayoff] = useState<PayoffInfo | null>(null);
  const [selectedBarrierType, setSelectedBarrierType] = useState<BarrierType>('None');
  const [parameters, setParameters] = useState<Record<string, any>>({});

  // Get available base payoffs for selected category
  const availableBasePayoffs = useMemo(() => {
    return getBasePayoffs().filter(p => p.category === selectedCategory);
  }, [selectedCategory]);

  // Get the final payoff based on selections
  const selectedPayoff = useMemo(() => {
    if (!selectedBasePayoff) return null;

    if (selectedBarrierType === 'None') {
      return selectedBasePayoff;
    }

    // Find the barrier variant
    const barrierPayoffName = `${selectedBarrierType}_${selectedBasePayoff.name}`;
    return ALL_PAYOFFS.find(p => p.name === barrierPayoffName) || null;
  }, [selectedBasePayoff, selectedBarrierType]);

  // Get all required parameters for the selected payoff
  const allParameters = useMemo(() => {
    if (!selectedPayoff) return [];
    return selectedPayoff.parameters;
  }, [selectedPayoff]);

  // Handle category change
  const handleCategoryChange = (category: PayoffCategory) => {
    setSelectedCategory(category);
    setSelectedBasePayoff(null);
    setSelectedBarrierType('None');
    setParameters({});
  };

  // Handle base payoff change
  const handleBasePayoffChange = (payoffName: string) => {
    const payoff = availableBasePayoffs.find(p => p.name === payoffName);
    setSelectedBasePayoff(payoff || null);
    setSelectedBarrierType('None');
    setParameters({});
  };

  // Handle barrier type change
  const handleBarrierTypeChange = (barrierType: BarrierType) => {
    setSelectedBarrierType(barrierType);
    // Clear barrier-specific parameters when changing barrier type
    const newParams = { ...parameters };
    getBarrierParameters(selectedBarrierType).forEach(param => {
      delete newParams[param.name];
    });
    setParameters(newParams);
  };

  // Handle parameter change
  const handleParameterChange = (paramName: string, value: any) => {
    setParameters(prev => ({
      ...prev,
      [paramName]: value,
    }));
  };

  // Handle submit
  const handleSubmit = () => {
    if (selectedPayoff && onPayoffSelect) {
      onPayoffSelect(selectedPayoff, parameters);
    }
  };

  return (
    <RetroPanel borderColor="cyan" title="Payoff Selector" className="w-full max-w-4xl mx-auto">
      <div className="space-y-6">
        {/* Category Selection */}
        <div>
          <RetroLabel>Category</RetroLabel>
          <RetroSelect
            value={selectedCategory}
            onChange={(e) => handleCategoryChange(e.target.value as PayoffCategory)}
            color="cyan"
          >
            <option value="Single Asset">Single Asset</option>
            <option value="Basket">Basket</option>
            <option value="Barrier Single Asset">Barrier Single Asset</option>
            <option value="Barrier Basket">Barrier Basket</option>
          </RetroSelect>
        </div>

        {/* Base Payoff Selection */}
        <div>
          <RetroLabel>Base Payoff</RetroLabel>
          <RetroSelect
            value={selectedBasePayoff?.name || ''}
            onChange={(e) => handleBasePayoffChange(e.target.value)}
            color="green"
            disabled={availableBasePayoffs.length === 0}
          >
            <option value="">Select a payoff...</option>
            {availableBasePayoffs.map(payoff => (
              <option key={payoff.name} value={payoff.name}>
                {payoff.name} ({payoff.abbreviation})
              </option>
            ))}
          </RetroSelect>
          {selectedBasePayoff && (
            <RetroDescription>{selectedBasePayoff.description}</RetroDescription>
          )}
        </div>

        {/* Barrier Type Selection */}
        {selectedBasePayoff && (
          <div>
            <RetroLabel>Barrier Type</RetroLabel>
            <RetroSelect
              value={selectedBarrierType}
              onChange={(e) => handleBarrierTypeChange(e.target.value as BarrierType)}
              color="magenta"
            >
              {BARRIER_TYPES.map(barrierType => (
                <option key={barrierType} value={barrierType}>
                  {barrierType === 'None' ? 'No Barrier' : barrierType}
                </option>
              ))}
            </RetroSelect>
            {selectedBarrierType !== 'None' && (
              <RetroDescription>{BARRIER_DESCRIPTIONS[selectedBarrierType]}</RetroDescription>
            )}
          </div>
        )}

        {/* Parameter Inputs */}
        {selectedPayoff && allParameters.length > 0 && (
          <div>
            <RetroLabel>Parameters</RetroLabel>
            <div className="space-y-4 mt-4">
              {allParameters.map(param => (
                <ParameterInput
                  key={param.name}
                  parameter={param}
                  value={parameters[param.name]}
                  onChange={(value) => handleParameterChange(param.name, value)}
                />
              ))}
            </div>
          </div>
        )}

        {/* Selected Payoff Summary */}
        {selectedPayoff && (
          <div className="mt-6">
            <RetroPanel borderColor="orange" title="Selected Payoff" noPadding>
              <div className="p-4 space-y-2">
                <div className="flex justify-between">
                  <span className="retro-text text-orange-400">Name:</span>
                  <span className="retro-text text-white">{selectedPayoff.name}</span>
                </div>
                <div className="flex justify-between">
                  <span className="retro-text text-orange-400">Abbreviation:</span>
                  <span className="retro-text text-white">{selectedPayoff.abbreviation}</span>
                </div>
                <div className="flex justify-between">
                  <span className="retro-text text-orange-400">Path Dependent:</span>
                  <span className="retro-text text-white">{selectedPayoff.isPathDependent ? 'Yes' : 'No'}</span>
                </div>
                <div className="flex justify-between">
                  <span className="retro-text text-orange-400">Multi-Asset:</span>
                  <span className="retro-text text-white">{selectedPayoff.requiresMultipleAssets ? 'Yes' : 'No'}</span>
                </div>
              </div>
            </RetroPanel>
          </div>
        )}

        {/* Submit Button */}
        {selectedPayoff && (
          <div className="flex justify-center mt-6">
            <RetroButton onClick={handleSubmit} variant="cyan">
              Apply Payoff
            </RetroButton>
          </div>
        )}
      </div>
    </RetroPanel>
  );
};

// ============================================================
// Helper Components
// ============================================================

interface RetroLabelProps {
  children: React.ReactNode;
}

const RetroLabel: React.FC<RetroLabelProps> = ({ children }) => (
  <label
    className="retro-text"
    style={{
      display: 'block',
      fontFamily: "'Press Start 2P', cursive",
      fontSize: '0.75rem',
      color: '#00ffff',
      textTransform: 'uppercase',
      letterSpacing: '1px',
      marginBottom: '0.5rem',
      textShadow: '0 0 5px #00ffff',
    }}
  >
    {children}
  </label>
);

interface RetroSelectProps extends React.SelectHTMLAttributes<HTMLSelectElement> {
  color?: 'cyan' | 'green' | 'magenta' | 'orange' | 'yellow';
}

const RetroSelect: React.FC<RetroSelectProps> = ({ color = 'cyan', children, ...props }) => {
  const colors = {
    cyan: '#00ffff',
    green: '#00ff00',
    magenta: '#ff00ff',
    orange: '#ff6600',
    yellow: '#ffff00',
  };

  const selectedColor = colors[color];

  return (
    <select
      {...props}
      style={{
        width: '100%',
        padding: '0.75rem',
        fontFamily: "'Press Start 2P', cursive",
        fontSize: '0.75rem',
        color: selectedColor,
        background: 'rgba(26, 26, 46, 0.9)',
        border: `2px solid ${selectedColor}`,
        borderRadius: '4px',
        boxShadow: `0 0 10px ${selectedColor}, inset 0 0 10px rgba(0, 0, 0, 0.5)`,
        outline: 'none',
        cursor: 'pointer',
        textShadow: `0 0 5px ${selectedColor}`,
        ...props.style,
      }}
      className={`retro-select ${props.className || ''}`}
    >
      {children}
    </select>
  );
};

interface RetroDescriptionProps {
  children: React.ReactNode;
}

const RetroDescription: React.FC<RetroDescriptionProps> = ({ children }) => (
  <p
    style={{
      fontFamily: "'Press Start 2P', cursive",
      fontSize: '0.6rem',
      color: '#888',
      marginTop: '0.5rem',
      lineHeight: '1.5',
      padding: '0.5rem',
      background: 'rgba(0, 255, 255, 0.05)',
      borderLeft: '2px solid #00ffff',
    }}
  >
    {children}
  </p>
);

interface ParameterInputProps {
  parameter: PayoffParameter;
  value: any;
  onChange: (value: any) => void;
}

const ParameterInput: React.FC<ParameterInputProps> = ({ parameter, value, onChange }) => {
  const getInputColor = (paramName: string): string => {
    if (paramName === 'strike') return '#00ff00';
    if (paramName.includes('barrier')) return '#ff00ff';
    if (paramName === 'k') return '#ffff00';
    if (paramName.includes('weight')) return '#ff6600';
    if (paramName.includes('step')) return '#00ffff';
    return '#00ffff';
  };

  const color = getInputColor(parameter.name);

  const renderInput = () => {
    if (parameter.type === 'number[]') {
      // Array input (for weights)
      return (
        <input
          type="text"
          value={value || ''}
          onChange={(e) => {
            try {
              const arr = e.target.value.split(',').map(v => parseFloat(v.trim())).filter(v => !isNaN(v));
              onChange(arr.length > 0 ? arr : undefined);
            } catch {
              onChange(undefined);
            }
          }}
          placeholder="e.g., 0.5, 0.3, 0.2"
          style={inputStyles(color)}
        />
      );
    } else if (parameter.type === 'integer') {
      // Integer input
      return (
        <input
          type="number"
          value={value || parameter.default || ''}
          onChange={(e) => onChange(parseInt(e.target.value) || parameter.default)}
          min={parameter.min}
          max={parameter.max}
          step={1}
          style={inputStyles(color)}
        />
      );
    } else {
      // Number input
      return (
        <input
          type="number"
          value={value !== undefined ? value : (parameter.default || '')}
          onChange={(e) => onChange(parseFloat(e.target.value) || parameter.default)}
          min={parameter.min}
          max={parameter.max}
          step={0.01}
          style={inputStyles(color)}
        />
      );
    }
  };

  return (
    <div>
      <label
        style={{
          display: 'block',
          fontFamily: "'Press Start 2P', cursive",
          fontSize: '0.65rem',
          color,
          textTransform: 'uppercase',
          letterSpacing: '1px',
          marginBottom: '0.5rem',
          textShadow: `0 0 5px ${color}`,
        }}
      >
        {parameter.name}
        {!parameter.required && (
          <span style={{ color: '#666', fontSize: '0.55rem' }}> (optional)</span>
        )}
      </label>
      {renderInput()}
      <p
        style={{
          fontFamily: "'Press Start 2P', cursive",
          fontSize: '0.55rem',
          color: '#666',
          marginTop: '0.25rem',
          lineHeight: '1.4',
        }}
      >
        {parameter.description}
        {parameter.default !== undefined && ` (default: ${parameter.default})`}
      </p>
    </div>
  );
};

const inputStyles = (color: string): React.CSSProperties => ({
  width: '100%',
  padding: '0.5rem',
  fontFamily: "'Press Start 2P', cursive",
  fontSize: '0.7rem',
  color,
  background: 'rgba(26, 26, 46, 0.9)',
  border: `2px solid ${color}`,
  borderRadius: '4px',
  boxShadow: `0 0 10px ${color}, inset 0 0 10px rgba(0, 0, 0, 0.5)`,
  outline: 'none',
  textShadow: `0 0 5px ${color}`,
});

interface RetroButtonProps {
  children: React.ReactNode;
  onClick?: () => void;
  variant?: 'cyan' | 'green' | 'magenta' | 'orange' | 'yellow';
  disabled?: boolean;
}

const RetroButton: React.FC<RetroButtonProps> = ({
  children,
  onClick,
  variant = 'cyan',
  disabled = false,
}) => {
  const [isHovered, setIsHovered] = useState(false);
  const [isPressed, setIsPressed] = useState(false);

  const colors = {
    cyan: '#00ffff',
    green: '#00ff00',
    magenta: '#ff00ff',
    orange: '#ff6600',
    yellow: '#ffff00',
  };

  const selectedColor = colors[variant];

  const buttonStyles: React.CSSProperties = {
    fontFamily: "'Press Start 2P', cursive",
    fontSize: '0.8rem',
    padding: '0.75rem 1.5rem',
    color: selectedColor,
    background: `rgba(${variant === 'cyan' ? '0, 255, 255' : variant === 'green' ? '0, 255, 0' : variant === 'magenta' ? '255, 0, 255' : variant === 'orange' ? '255, 102, 0' : '255, 255, 0'}, 0.1)`,
    border: `3px solid ${selectedColor}`,
    borderRadius: '4px',
    boxShadow: isHovered && !disabled
      ? `0 0 20px ${selectedColor}, 0 0 30px ${selectedColor}, inset 0 0 15px rgba(0, 0, 0, 0.3)`
      : `0 0 10px ${selectedColor}, inset 0 0 10px rgba(0, 0, 0, 0.2)`,
    textShadow: `0 0 5px ${selectedColor}`,
    textTransform: 'uppercase',
    letterSpacing: '1px',
    cursor: disabled ? 'not-allowed' : 'pointer',
    transition: 'all 0.3s ease',
    transform: isPressed && !disabled
      ? 'scale(0.95)'
      : isHovered && !disabled
      ? 'scale(1.05)'
      : 'scale(1)',
    opacity: disabled ? 0.5 : 1,
  };

  return (
    <button
      onClick={disabled ? undefined : onClick}
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => {
        setIsHovered(false);
        setIsPressed(false);
      }}
      onMouseDown={() => setIsPressed(true)}
      onMouseUp={() => setIsPressed(false)}
      disabled={disabled}
      style={buttonStyles}
    >
      {children}
    </button>
  );
};

export default PayoffSelector;
