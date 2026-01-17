# MixInto Frontend

React + TypeScript + Tailwind frontend for the MixInto audio extension tool.

## Development

1. Install dependencies:
```bash
npm install
```

2. Start the development server:
```bash
npm run dev
```

The frontend will run on `http://localhost:5173` and proxy API requests to the Flask backend.

## Building for Production

```bash
npm run build
```

This creates a `dist` folder that the Flask backend will serve in production.

## Project Structure

- `src/App.tsx` - Main app component with tab navigation
- `src/components/AnalyzeTab.tsx` - Analyze form component
- `src/components/ExtendTab.tsx` - Extend form component
- `src/main.tsx` - Entry point
- `src/index.css` - Tailwind CSS imports
