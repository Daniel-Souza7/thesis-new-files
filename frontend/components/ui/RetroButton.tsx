import React from 'react';

interface RetroButtonProps {
  children: React.ReactNode;
  onClick?: () => void;
  variant?: 'orange' | 'green' | 'magenta' | 'cyan' | 'yellow';
  size?: 'sm' | 'md' | 'lg';
  disabled?: boolean;
  className?: string;
  type?: 'button' | 'submit' | 'reset';
}

export const RetroButton: React.FC<RetroButtonProps> = ({
  children,
  onClick,
  variant = 'orange',
  size = 'md',
  disabled = false,
  className = '',
  type = 'button',
}) => {
  const variantStyles = {
    orange: {
      color: '#ff6600',
      background: 'rgba(255, 102, 0, 0.1)',
      border: '3px solid #ff6600',
      boxShadow: '0 0 10px #ff6600, inset 0 0 10px rgba(255, 102, 0, 0.2)',
      hoverBoxShadow: '0 0 20px #ff6600, 0 0 30px #ff6600, inset 0 0 15px rgba(255, 102, 0, 0.3)',
    },
    green: {
      color: '#00ff00',
      background: 'rgba(0, 255, 0, 0.1)',
      border: '3px solid #00ff00',
      boxShadow: '0 0 10px #00ff00, inset 0 0 10px rgba(0, 255, 0, 0.2)',
      hoverBoxShadow: '0 0 20px #00ff00, 0 0 30px #00ff00, inset 0 0 15px rgba(0, 255, 0, 0.3)',
    },
    magenta: {
      color: '#ff00ff',
      background: 'rgba(255, 0, 255, 0.1)',
      border: '3px solid #ff00ff',
      boxShadow: '0 0 10px #ff00ff, inset 0 0 10px rgba(255, 0, 255, 0.2)',
      hoverBoxShadow: '0 0 20px #ff00ff, 0 0 30px #ff00ff, inset 0 0 15px rgba(255, 0, 255, 0.3)',
    },
    cyan: {
      color: '#00ffff',
      background: 'rgba(0, 255, 255, 0.1)',
      border: '3px solid #00ffff',
      boxShadow: '0 0 10px #00ffff, inset 0 0 10px rgba(0, 255, 255, 0.2)',
      hoverBoxShadow: '0 0 20px #00ffff, 0 0 30px #00ffff, inset 0 0 15px rgba(0, 255, 255, 0.3)',
    },
    yellow: {
      color: '#ffff00',
      background: 'rgba(255, 255, 0, 0.1)',
      border: '3px solid #ffff00',
      boxShadow: '0 0 10px #ffff00, inset 0 0 10px rgba(255, 255, 0, 0.2)',
      hoverBoxShadow: '0 0 20px #ffff00, 0 0 30px #ffff00, inset 0 0 15px rgba(255, 255, 0, 0.3)',
    },
  };

  const sizeStyles = {
    sm: {
      padding: '0.5rem 1rem',
      fontSize: '0.7rem',
    },
    md: {
      padding: '0.75rem 1.5rem',
      fontSize: '0.9rem',
    },
    lg: {
      padding: '1rem 2rem',
      fontSize: '1.1rem',
    },
  };

  const selectedVariant = variantStyles[variant];
  const selectedSize = sizeStyles[size];

  const baseStyles: React.CSSProperties = {
    fontFamily: "'Press Start 2P', cursive",
    textTransform: 'uppercase',
    letterSpacing: '1px',
    cursor: disabled ? 'not-allowed' : 'pointer',
    transition: 'all 0.3s ease',
    position: 'relative',
    overflow: 'hidden',
    userSelect: 'none',
    whiteSpace: 'nowrap',
    ...selectedSize,
    ...selectedVariant,
    opacity: disabled ? 0.5 : 1,
  };

  const [isHovered, setIsHovered] = React.useState(false);
  const [isPressed, setIsPressed] = React.useState(false);

  const buttonStyles: React.CSSProperties = {
    ...baseStyles,
    boxShadow: isHovered && !disabled
      ? selectedVariant.hoverBoxShadow
      : selectedVariant.boxShadow,
    transform: isPressed && !disabled
      ? 'scale(0.95)'
      : isHovered && !disabled
      ? 'scale(1.05)'
      : 'scale(1)',
    textShadow: `0 0 5px ${selectedVariant.color}`,
  };

  return (
    <button
      type={type}
      onClick={disabled ? undefined : onClick}
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => {
        setIsHovered(false);
        setIsPressed(false);
      }}
      onMouseDown={() => setIsPressed(true)}
      onMouseUp={() => setIsPressed(false)}
      disabled={disabled}
      className={className}
      style={buttonStyles}
    >
      {children}
    </button>
  );
};

export default RetroButton;
