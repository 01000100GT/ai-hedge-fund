```json
{
  "name": "vite-react-flow-template", // 项目名称
  "version": "0.0.0", // 版本
  "type": "module", // 模块类型
  "scripts": { // 脚本
    "dev": "vite", // 开发模式
    "build": "tsc && vite build", // 构建生产版本
    "lint": "eslint . --ext ts,tsx --report-unused-disable-directives --max-warnings 0", // 代码风格检查
    "preview": "vite preview" // 预览生产版本
  },
  "dependencies": { // 生产依赖
    "@radix-ui/react-accordion": "^1.2.10", // Radix UI - 手风琴组件
    "@radix-ui/react-checkbox": "^1.3.2", // Radix UI - 复选框组件
    "@radix-ui/react-dialog": "^1.1.13", // Radix UI - 对话框组件
    "@radix-ui/react-icons": "^1.3.2", // Radix UI - 图标库
    "@radix-ui/react-popover": "^1.1.13", // Radix UI - 气泡卡片组件
    "@radix-ui/react-separator": "^1.1.6", // Radix UI - 分隔符组件
    "@radix-ui/react-slot": "^1.2.0", // Radix UI - 插槽组件
    "@radix-ui/react-tabs": "^1.1.11", // Radix UI - 标签页组件
    "@radix-ui/react-tooltip": "^1.2.6", // Radix UI - 工具提示组件
    "@types/react-syntax-highlighter": "^15.5.13", // TypeScript 类型定义 - React 代码高亮
    "@xyflow/react": "^12.5.1", // React Flow 库
    "class-variance-authority": "^0.7.1", // 类变体工具
    "clsx": "^2.1.1", // 条件合并 CSS 类名工具
    "cmdk": "^1.1.1", // 命令面板 UI
    "lucide-react": "^0.507.0", // Lucide 图标库 - React
    "next-themes": "^0.4.6", // Next.js 主题切换
    "react": "^18.2.0", // React 核心库
    "react-dom": "^18.2.0", // React DOM
    "react-resizable-panels": "^3.0.1", // 可调整大小的面板组件
    "react-syntax-highlighter": "^15.6.1", // React 代码高亮组件
    "shadcn-ui": "^0.9.5", // Shadcn UI 组件库
    "sonner": "^2.0.5", // Toast 通知组件
    "tailwind-merge": "^3.2.0" // Tailwind CSS 类名合并工具
  },
  "license": "MIT", // 许可证
  "devDependencies": { // 开发依赖
    "@tailwindcss/typography": "^0.5.16", // Tailwind CSS 排版插件
    "@types/node": "^22.15.3", // TypeScript 类型定义 - Node.js
    "@types/react": "^18.2.53", // TypeScript 类型定义 - React
    "@types/react-dom": "^18.2.18", // TypeScript 类型定义 - React DOM
    "@typescript-eslint/eslint-plugin": "^6.20.0", // ESLint TypeScript 插件
    "@typescript-eslint/parser": "^6.20.0", // ESLint TypeScript 解析器
    "@vitejs/plugin-react": "^4.2.1", // Vite React 插件
    "autoprefixer": "^10.4.21", // 自动添加 CSS 厂商前缀
    "eslint": "^8.56.0", // JavaScript 代码检查工具
    "eslint-plugin-react-hooks": "^4.6.0", // ESLint React Hooks 插件
    "eslint-plugin-react-refresh": "^0.4.5", // ESLint React Refresh 插件
    "postcss": "^8.5.3", // CSS 后处理器
    "tailwindcss": "^3.4.1", // 实用优先的 CSS 框架
    "tailwindcss-animate": "^1.0.7", // Tailwind CSS 动画插件
    "typescript": "^5.3.3", // TypeScript 语言
    "vite": "^5.0.12" // 下一代前端开发工具
  }
} 