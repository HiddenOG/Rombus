import React, { useState } from 'react';
import { ErrorBoundary } from './components/ErrorBoundary';
import { Sidebar } from './components/Sidebar';
import { Notebook } from './components/Notebook';
import { useNotebook } from './hooks/useNotebook';
import { useAccount } from './hooks/useAccount';

const SIDEBAR_KEY = 'automl.sidebar';
const isDesktop = () => window.matchMedia('(min-width: 1024px)').matches;

function initialSidebar() {
  // Phones start with the drawer closed; computers remember the last choice (open by default)
  if (!isDesktop()) return false;
  try {
    return localStorage.getItem(SIDEBAR_KEY) !== 'closed';
  } catch {
    return true;
  }
}

/**
 * App — sidebar (notebooks + dataset) and the notebook of cells.
 */
const App = () => {
  const nb = useNotebook();
  const account = useAccount({
    onSignedIn: nb.refreshNotebooks,
    onSignedOut: () => {
      nb.newNotebook();
      nb.refreshNotebooks();
    },
  });
  const [sidebarOpen, setSidebarOpen] = useState(initialSidebar);

  const toggleSidebar = () => {
    setSidebarOpen(open => {
      if (isDesktop()) {
        try {
          localStorage.setItem(SIDEBAR_KEY, open ? 'closed' : 'open');
        } catch {
          // Storage blocked: the choice just won't be remembered
        }
      }
      return !open;
    });
  };
  // On phones the drawer covers the notebook, so close it once a choice is made
  const thenCloseOnPhone = (fn) => (...args) => {
    if (!isDesktop()) setSidebarOpen(false);
    return fn(...args);
  };

  return (
    <ErrorBoundary>
      <div className="flex h-dvh overflow-hidden bg-canvas font-sans text-ink">
        <Sidebar
          open={sidebarOpen}
          onClose={toggleSidebar}
          sessionId={nb.sessionId}
          summary={nb.summary}
          onFileUpload={thenCloseOnPhone(nb.upload)}
          isProcessing={nb.isProcessing}
          notebooks={nb.notebooks}
          onOpenNotebook={thenCloseOnPhone(nb.openNotebook)}
          onNewNotebook={nb.newNotebook}
          onDeleteNotebook={nb.deleteNotebook}
          account={account}
        />
        <Notebook nb={nb} signedIn={Boolean(account.user)} sidebarOpen={sidebarOpen} onToggleSidebar={toggleSidebar} />
      </div>
    </ErrorBoundary>
  );
};

export default App;
