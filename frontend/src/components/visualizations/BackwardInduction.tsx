import { useState, useEffect, useRef } from 'react';
import { motion } from 'framer-motion';
import * as d3 from 'd3';

interface TreeNode {
  time: number;
  price: number;
  value: number;
  exercise: boolean;
  children?: TreeNode[];
}

const BackwardInduction = () => {
  const svgRef = useRef<SVGSVGElement>(null);
  const [currentStep, setCurrentStep] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [treeData, setTreeData] = useState<TreeNode | null>(null);
  const totalSteps = 4; // 0, 1, 2, 3 time steps

  useEffect(() => {
    generateTreeData();
  }, []);

  useEffect(() => {
    if (treeData) {
      drawTree();
    }
  }, [treeData, currentStep]);

  const generateTreeData = () => {
    const S0 = 100;
    const K = 100;
    const u = 1.2;
    const d = 0.8;
    const r = 0.05;
    const dt = 1 / 3;

    // Build binomial tree (simplified to 3 time steps)
    const tree: TreeNode = {
      time: 0,
      price: S0,
      value: 0,
      exercise: false,
      children: [
        {
          time: 1,
          price: S0 * u,
          value: 0,
          exercise: false,
          children: [
            {
              time: 2,
              price: S0 * u * u,
              value: 0,
              exercise: false,
              children: [
                {
                  time: 3,
                  price: S0 * u * u * u,
                  value: Math.max(K - S0 * u * u * u, 0),
                  exercise: false,
                },
                {
                  time: 3,
                  price: S0 * u * u * d,
                  value: Math.max(K - S0 * u * u * d, 0),
                  exercise: false,
                },
              ],
            },
            {
              time: 2,
              price: S0 * u * d,
              value: 0,
              exercise: false,
              children: [
                {
                  time: 3,
                  price: S0 * u * d * u,
                  value: Math.max(K - S0 * u * d * u, 0),
                  exercise: false,
                },
                {
                  time: 3,
                  price: S0 * u * d * d,
                  value: Math.max(K - S0 * u * d * d, 0),
                  exercise: false,
                },
              ],
            },
          ],
        },
        {
          time: 1,
          price: S0 * d,
          value: 0,
          exercise: false,
          children: [
            {
              time: 2,
              price: S0 * d * u,
              value: 0,
              exercise: false,
              children: [
                {
                  time: 3,
                  price: S0 * d * u * u,
                  value: Math.max(K - S0 * d * u * u, 0),
                  exercise: false,
                },
                {
                  time: 3,
                  price: S0 * d * u * d,
                  value: Math.max(K - S0 * d * u * d, 0),
                  exercise: false,
                },
              ],
            },
            {
              time: 2,
              price: S0 * d * d,
              value: 0,
              exercise: false,
              children: [
                {
                  time: 3,
                  price: S0 * d * d * u,
                  value: Math.max(K - S0 * d * d * u, 0),
                  exercise: false,
                },
                {
                  time: 3,
                  price: S0 * d * d * d,
                  value: Math.max(K - S0 * d * d * d, 0),
                  exercise: false,
                },
              ],
            },
          ],
        },
      ],
    };

    // Calculate backward induction values
    const calculateValues = (node: TreeNode, step: number): number => {
      if (node.time === 3) {
        return node.value;
      }

      if (node.children && node.time < step) {
        const discount = Math.exp(-r * dt);
        const continuation = discount * 0.5 * (
          calculateValues(node.children[0], step) + calculateValues(node.children[1], step)
        );
        const immediate = Math.max(K - node.price, 0);
        node.value = Math.max(continuation, immediate);
        node.exercise = immediate >= continuation;
        return node.value;
      }

      return node.value;
    };

    // Calculate values for current step
    if (currentStep > 0) {
      calculateValues(tree, 4 - currentStep);
    }

    setTreeData(tree);
  };

  const drawTree = () => {
    if (!svgRef.current || !treeData) return;

    const width = 700;
    const height = 500;
    const margin = { top: 40, right: 40, bottom: 40, left: 40 };

    // Clear previous content
    d3.select(svgRef.current).selectAll('*').remove();

    const svg = d3.select(svgRef.current)
      .attr('width', width)
      .attr('height', height);

    const g = svg.append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`);

    const treeLayout = d3.tree<TreeNode>()
      .size([height - margin.top - margin.bottom, width - margin.left - margin.right]);

    const root = d3.hierarchy(treeData, (d) => d.children);
    const treeNodes = treeLayout(root);

    // Draw links
    g.selectAll('.link')
      .data(treeNodes.links())
      .enter()
      .append('line')
      .attr('class', 'link')
      .attr('x1', (d) => d.source.y)
      .attr('y1', (d) => d.source.x)
      .attr('x2', (d) => d.target.y)
      .attr('y2', (d) => d.target.x)
      .attr('stroke', '#cbd5e1')
      .attr('stroke-width', 2);

    // Draw nodes
    const nodes = g.selectAll('.node')
      .data(treeNodes.descendants())
      .enter()
      .append('g')
      .attr('class', 'node')
      .attr('transform', (d) => `translate(${d.y},${d.x})`);

    // Determine node colors based on current step
    nodes.append('circle')
      .attr('r', 25)
      .attr('fill', (d) => {
        const node = d.data;
        const stepFromEnd = 3 - currentStep;
        if (node.time > stepFromEnd) return '#94a3b8'; // Future (gray)
        if (node.exercise) return '#ef4444'; // Exercise (red)
        return '#3b82f6'; // Hold (blue)
      })
      .attr('stroke', '#1e293b')
      .attr('stroke-width', 2);

    // Add price labels
    nodes.append('text')
      .attr('dy', -35)
      .attr('text-anchor', 'middle')
      .attr('fill', '#1e293b')
      .attr('font-size', '12px')
      .attr('font-weight', 'bold')
      .text((d) => `$${d.data.price.toFixed(0)}`);

    // Add value labels
    nodes.append('text')
      .attr('dy', 5)
      .attr('text-anchor', 'middle')
      .attr('fill', 'white')
      .attr('font-size', '11px')
      .attr('font-weight', 'bold')
      .text((d) => {
        const stepFromEnd = 3 - currentStep;
        if (d.data.time > stepFromEnd) return '';
        return d.data.value.toFixed(2);
      });

    // Add time step labels
    const timeSteps = [0, 1, 2, 3];
    g.selectAll('.time-label')
      .data(timeSteps)
      .enter()
      .append('text')
      .attr('class', 'time-label')
      .attr('x', (d) => (width - margin.left - margin.right) * d / 3)
      .attr('y', -20)
      .attr('text-anchor', 'middle')
      .attr('fill', '#64748b')
      .attr('font-size', '14px')
      .attr('font-weight', 'bold')
      .text((d) => `t=${d}`);
  };

  useEffect(() => {
    generateTreeData();
  }, [currentStep]);

  const play = () => {
    setIsPlaying(true);
    let step = 0;
    const interval = setInterval(() => {
      step++;
      setCurrentStep(step);
      if (step >= totalSteps) {
        clearInterval(interval);
        setIsPlaying(false);
      }
    }, 1500);
  };

  const reset = () => {
    setCurrentStep(0);
    setIsPlaying(false);
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="w-full bg-white rounded-lg shadow-lg p-6"
    >
      <h3 className="text-xl font-semibold mb-2">Backward Induction Animation</h3>
      <p className="text-gray-600 mb-4">
        Step through the dynamic programming process for an American put option using a binomial tree
      </p>

      {/* Option and tree specification */}
      <div className="mb-4 p-3 bg-gradient-to-r from-green-50 to-emerald-50 rounded-lg border border-green-200">
        <p className="text-sm text-gray-700">
          <strong>Option:</strong> American Put | <strong>Strike:</strong> K=$100 | <strong>Initial Price:</strong> S₀=$100 |
          <strong> Up factor:</strong> u=1.2 | <strong>Down factor:</strong> d=0.8 | <strong>Steps:</strong> 3 time periods
        </p>
        <p className="text-xs text-gray-600 mt-1">
          Each circle shows the stock price (above) and option value (inside). The tree represents all possible price paths.
        </p>
      </div>

      <div className="flex gap-3 mb-6">
        <button
          onClick={play}
          disabled={isPlaying || currentStep === totalSteps}
          className="px-4 py-2 bg-blue-600 text-white rounded-lg font-medium hover:bg-blue-700 transition-colors disabled:bg-gray-400 disabled:cursor-not-allowed"
        >
          {isPlaying ? 'Playing...' : 'Play'}
        </button>
        <button
          onClick={reset}
          disabled={isPlaying}
          className="px-4 py-2 bg-gray-600 text-white rounded-lg font-medium hover:bg-gray-700 transition-colors disabled:bg-gray-400 disabled:cursor-not-allowed"
        >
          Reset
        </button>
        <div className="flex-1"></div>
        <div className="text-right">
          <p className="text-sm text-gray-600">Current Step</p>
          <p className="text-2xl font-bold text-blue-600">{currentStep} / {totalSteps}</p>
        </div>
      </div>

      <div className="flex justify-center mb-4">
        <svg ref={svgRef}></svg>
      </div>

      <div className="grid grid-cols-3 gap-4 mt-4">
        <div className="flex items-center gap-2">
          <div className="w-6 h-6 rounded-full bg-blue-600 border-2 border-gray-900"></div>
          <span className="text-sm text-gray-700">Hold (continuation value higher)</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-6 h-6 rounded-full bg-red-600 border-2 border-gray-900"></div>
          <span className="text-sm text-gray-700">Exercise (immediate payoff higher)</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-6 h-6 rounded-full bg-gray-400 border-2 border-gray-900"></div>
          <span className="text-sm text-gray-700">Not yet calculated</span>
        </div>
      </div>

      <div className="mt-4 p-4 bg-blue-50 rounded-lg">
        <p className="text-sm text-gray-700">
          <strong>Algorithm:</strong> Starting from maturity (t=3), we work backward. At each node,
          we compare the immediate exercise value max(K-S, 0) with the discounted expected
          continuation value. The option value is the maximum of these two.
        </p>
      </div>
    </motion.div>
  );
};

export default BackwardInduction;
