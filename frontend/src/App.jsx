import React from 'react';
import { ErrorBoundary } from './components/ErrorBoundary';
import { Sidebar } from './components/Sidebar';
import { Notebook } from './components/Notebook';
import { useNotebook } from './hooks/useNotebook';

/**
 * App — sidebar (notebooks + dataset) and the notebook of cells.
 */
const App = () => {
  const nb = useNotebook();

  return (
    <ErrorBoundary>
      <div className="flex h-screen overflow-hidden bg-canvas font-sans text-ink">
        <Sidebar
          sessionId={nb.sessionId}
          summary={nb.summary}
          onFileUpload={nb.upload}
          isProcessing={nb.isProcessing}
          notebooks={nb.notebooks}
          onOpenNotebook={nb.openNotebook}
          onNewNotebook={nb.newNotebook}
          onDeleteNotebook={nb.deleteNotebook}
        />
        <Notebook nb={nb} />
      </div>
    </ErrorBoundary>
  );
};

export default App;
