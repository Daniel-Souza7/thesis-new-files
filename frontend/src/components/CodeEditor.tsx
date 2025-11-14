import { useState, useEffect } from 'react';
import Editor from '@monaco-editor/react';
import { motion } from 'framer-motion';
import { Copy, Check, Code2 } from 'lucide-react';

interface CodeEditorProps {
  title?: string;
  language?: string;
  defaultCode: string;
  height?: string;
  readOnly?: boolean;
}

const CodeEditor = ({
  title = 'Code Example',
  language = 'python',
  defaultCode,
  height = '400px',
  readOnly = true,
}: CodeEditorProps) => {
  const [code, setCode] = useState(defaultCode);
  const [copied, setCopied] = useState(false);

  // Update code when defaultCode changes
  useEffect(() => {
    setCode(defaultCode);
  }, [defaultCode]);

  const handleCopy = () => {
    navigator.clipboard.writeText(code);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="w-full bg-white rounded-lg shadow-lg overflow-hidden"
    >
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 bg-gray-900 border-b border-gray-700">
        <div className="flex items-center gap-2">
          <Code2 className="w-5 h-5 text-blue-400" />
          <span className="text-sm font-medium text-gray-200">{title}</span>
          <span className="text-xs text-gray-400 px-2 py-1 bg-gray-800 rounded">
            {language}
          </span>
        </div>
        <button
          onClick={handleCopy}
          className="flex items-center gap-2 px-3 py-1.5 text-sm text-gray-300 hover:text-white hover:bg-gray-800 rounded transition-colors"
        >
          {copied ? (
            <>
              <Check className="w-4 h-4" />
              Copied!
            </>
          ) : (
            <>
              <Copy className="w-4 h-4" />
              Copy
            </>
          )}
        </button>
      </div>

      {/* Editor */}
      <Editor
        height={height}
        language={language}
        value={code}
        onChange={(value) => setCode(value || '')}
        theme="vs-dark"
        options={{
          readOnly,
          minimap: { enabled: false },
          fontSize: 14,
          lineNumbers: 'on',
          scrollBeyondLastLine: false,
          wordWrap: 'on',
          automaticLayout: true,
          padding: { top: 16, bottom: 16 },
        }}
      />
    </motion.div>
  );
};

export default CodeEditor;
