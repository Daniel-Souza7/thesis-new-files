import React from 'react';

interface ArcadeCabinetProps {
  children: React.ReactNode;
  title?: string;
  className?: string;
}

export const ArcadeCabinet: React.FC<ArcadeCabinetProps> = ({
  children,
  title = 'ARCADE',
  className = '',
}) => {
  const cabinetStyles: React.CSSProperties = {
    maxWidth: '1400px',
    margin: '2rem auto',
    padding: '0',
    position: 'relative',
    borderRadius: '16px',
    overflow: 'hidden',
  };

  const topBarStyles: React.CSSProperties = {
    background: '#00ff00',
    padding: '1.5rem',
    textAlign: 'center',
    borderBottom: '4px solid #00ff00',
    boxShadow: '0 0 20px rgba(0, 255, 0, 0.8)',
    position: 'relative',
  };

  const titleStyles: React.CSSProperties = {
    fontFamily: "'Press Start 2P', cursive",
    fontSize: 'clamp(1.5rem, 4vw, 2.5rem)',
    color: '#0f0c29',
    textTransform: 'uppercase',
    letterSpacing: '4px',
    margin: 0,
    textShadow: '2px 2px 4px rgba(0, 0, 0, 0.8)',
    fontWeight: 'bold',
  };

  const bodyStyles: React.CSSProperties = {
    background: 'rgba(15, 12, 41, 0.95)',
    border: '6px solid #00ff00',
    borderTop: 'none',
    borderRadius: '0 0 16px 16px',
    boxShadow: `
      0 0 30px #00ff00,
      0 0 60px rgba(0, 255, 0, 0.5),
      inset 0 0 30px rgba(0, 0, 0, 0.7)
    `,
    padding: '2rem',
    minHeight: '400px',
    position: 'relative',
  };

  const cornerDecorStyles: React.CSSProperties = {
    position: 'absolute',
    width: '20px',
    height: '20px',
    border: '3px solid #00ff00',
    boxShadow: '0 0 10px #00ff00',
  };

  return (
    <div style={cabinetStyles} className={className}>
      {/* Top Bar */}
      <div style={topBarStyles}>
        <h1 style={titleStyles}>{title}</h1>
      </div>

      {/* Main Body */}
      <div style={bodyStyles}>
        {/* Corner Decorations */}
        <div style={{ ...cornerDecorStyles, top: '10px', left: '10px', borderRight: 'none', borderBottom: 'none' }} />
        <div style={{ ...cornerDecorStyles, top: '10px', right: '10px', borderLeft: 'none', borderBottom: 'none' }} />
        <div style={{ ...cornerDecorStyles, bottom: '10px', left: '10px', borderRight: 'none', borderTop: 'none' }} />
        <div style={{ ...cornerDecorStyles, bottom: '10px', right: '10px', borderLeft: 'none', borderTop: 'none' }} />

        {/* Content */}
        {children}
      </div>
    </div>
  );
};

export default ArcadeCabinet;
