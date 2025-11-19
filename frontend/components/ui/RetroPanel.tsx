import React from 'react';

interface RetroPanelProps {
  children: React.ReactNode;
  borderColor?: 'green' | 'cyan' | 'magenta' | 'yellow' | 'orange';
  title?: string;
  className?: string;
  noPadding?: boolean;
}

export const RetroPanel: React.FC<RetroPanelProps> = ({
  children,
  borderColor = 'cyan',
  title,
  className = '',
  noPadding = false,
}) => {
  const borderColors = {
    green: '#00ff00',
    cyan: '#00ffff',
    magenta: '#ff00ff',
    yellow: '#ffff00',
    orange: '#ff6600',
  };

  const selectedColor = borderColors[borderColor];

  const panelStyles: React.CSSProperties = {
    background: 'rgba(26, 26, 46, 0.9)',
    border: `3px solid ${selectedColor}`,
    borderRadius: '8px',
    boxShadow: `
      0 0 15px ${selectedColor},
      inset 0 0 15px rgba(0, 0, 0, 0.5)
    `,
    padding: noPadding ? '0' : '1.5rem',
    position: 'relative',
    backdropFilter: 'blur(5px)',
  };

  const titleStyles: React.CSSProperties = {
    fontFamily: "'Press Start 2P', cursive",
    fontSize: '1rem',
    color: selectedColor,
    textTransform: 'uppercase',
    letterSpacing: '2px',
    marginBottom: '1rem',
    textShadow: `
      0 0 10px ${selectedColor},
      0 0 20px ${selectedColor}
    `,
    paddingBottom: '0.5rem',
    borderBottom: `2px solid ${selectedColor}`,
  };

  return (
    <div style={panelStyles} className={className}>
      {title && <div style={titleStyles}>{title}</div>}
      {children}
    </div>
  );
};

export default RetroPanel;
