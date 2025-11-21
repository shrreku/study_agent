export function PrimaryButton(props) {
  const { children, style, ...rest } = props
  return (
    <button
      type="button"
      {...rest}
      style={{
        padding: '8px 14px',
        borderRadius: 999,
        border: '1px solid #2563eb',
        background: '#2563eb',
        color: '#f9fafb',
        fontSize: 14,
        fontWeight: 500,
        cursor: 'pointer',
        display: 'inline-flex',
        alignItems: 'center',
        justifyContent: 'center',
        gap: 6,
        ...style,
      }}
    >
      {children}
    </button>
  )
}

export function SecondaryButton(props) {
  const { children, style, ...rest } = props
  return (
    <button
      type="button"
      {...rest}
      style={{
        padding: '8px 14px',
        borderRadius: 999,
        border: '1px solid #d1d5db',
        background: '#ffffff',
        color: '#374151',
        fontSize: 14,
        fontWeight: 500,
        cursor: 'pointer',
        display: 'inline-flex',
        alignItems: 'center',
        justifyContent: 'center',
        gap: 6,
        ...style,
      }}
    >
      {children}
    </button>
  )
}

export function Card({ title, subtitle, children, style, footer, ...rest }) {
  return (
    <section
      {...rest}
      style={{
        background: '#ffffff',
        borderRadius: 12,
        padding: 16,
        boxShadow: '0 1px 3px rgba(15, 23, 42, 0.06)',
        border: '1px solid #e5e7eb',
        ...style,
      }}
    >
      {(title || subtitle) && (
        <header style={{ marginBottom: 8 }}>
          {title && <h2 style={{ fontSize: 18, margin: 0 }}>{title}</h2>}
          {subtitle && (
            <p style={{ margin: '4px 0 0', fontSize: 13, color: '#6b7280' }}>{subtitle}</p>
          )}
        </header>
      )}
      <div>{children}</div>
      {footer && <footer style={{ marginTop: 12 }}>{footer}</footer>}
    </section>
  )
}

export function InputField({ label, error, helper, ...rest }) {
  return (
    <div style={{ marginBottom: 12 }}>
      {label && (
        <label style={{ display: 'block', marginBottom: 4, fontSize: 13, color: '#374151' }}>
          {label}
        </label>
      )}
      <input
        {...rest}
        style={{
          width: '100%',
          padding: '7px 10px',
          borderRadius: 8,
          border: `1px solid ${error ? '#ef4444' : '#d1d5db'}`,
          fontSize: 14,
        }}
      />
      {helper && !error && (
        <div style={{ marginTop: 4, fontSize: 12, color: '#6b7280' }}>{helper}</div>
      )}
      {error && <div style={{ marginTop: 4, fontSize: 12, color: '#b91c1c' }}>{error}</div>}
    </div>
  )
}

export function ProgressBar({ value, label }) {
  const clamped = Math.max(0, Math.min(100, Number.isFinite(value) ? value : 0))
  return (
    <div style={{ width: '100%' }}>
      <div
        style={{
          height: 8,
          borderRadius: 999,
          background: '#e5e7eb',
          overflow: 'hidden',
        }}
      >
        <div
          style={{
            width: `${clamped}%`,
            height: '100%',
            background: '#2563eb',
            transition: 'width 0.2s ease-out',
          }}
        />
      </div>
      {label && (
        <div style={{ marginTop: 4, fontSize: 12, color: '#6b7280' }}>{label}</div>
      )}
    </div>
  )
}
