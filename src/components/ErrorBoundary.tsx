import React from 'react'

interface ErrorBoundaryProps {
  children: React.ReactNode
}

interface ErrorBoundaryState {
  error: Error | null
}

/**
 * Keeps a component crash from blanking the whole window. Without this, any
 * render error (for example a WebGL viewer failing on an unsupported webview)
 * leaves the user staring at a black screen with no way back.
 */
export class ErrorBoundary extends React.Component<ErrorBoundaryProps, ErrorBoundaryState> {
  constructor(props: ErrorBoundaryProps) {
    super(props)
    this.state = { error: null }
  }

  static getDerivedStateFromError(error: Error): ErrorBoundaryState {
    return { error }
  }

  componentDidCatch(error: Error, info: React.ErrorInfo) {
    console.error('[ErrorBoundary]', error, info.componentStack)
  }

  private reset = () => {
    this.setState({ error: null })
  }

  render() {
    const { error } = this.state
    if (!error) return this.props.children

    return (
      <div
        style={{
          minHeight: '100vh',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          background: '#0b0e13',
          color: '#e6e9ef',
          padding: 24,
          fontFamily: 'ui-monospace, SFMono-Regular, Menlo, monospace',
        }}
      >
        <div style={{ maxWidth: 520, width: '100%', border: '1px solid #2a2f3a', borderRadius: 6, padding: 20, background: '#11151c' }}>
          <p style={{ fontSize: 12, letterSpacing: '0.08em', textTransform: 'uppercase', color: '#8b93a3', margin: 0 }}>
            view error
          </p>
          <p style={{ fontSize: 13, marginTop: 10, marginBottom: 16, lineHeight: 1.5 }}>
            This panel could not be rendered. The rest of the session is safe.
          </p>
          <pre
            style={{
              fontSize: 11,
              background: '#0b0e13',
              border: '1px solid #232833',
              borderRadius: 4,
              padding: 10,
              overflow: 'auto',
              maxHeight: 160,
              color: '#b9c0cf',
              margin: 0,
            }}
          >
            {error.message}
          </pre>
          <button
            onClick={this.reset}
            style={{
              marginTop: 16,
              fontSize: 11,
              padding: '6px 12px',
              background: '#1b2430',
              color: '#e6e9ef',
              border: '1px solid #2a2f3a',
              borderRadius: 4,
              cursor: 'pointer',
            }}
          >
            try again
          </button>
        </div>
      </div>
    )
  }
}
