import { NavBar } from './components/NavBar'

export const metadata = {
  title: 'StudyAgent',
}

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body
        style={{
          margin: 0,
          fontFamily: '-apple-system, BlinkMacSystemFont, system-ui, sans-serif',
          background: '#f3f4f6',
          color: '#111827',
        }}
      >
        <NavBar />
        <div style={{ maxWidth: 960, margin: '0 auto', padding: '24px 16px 40px' }}>{children}</div>
      </body>
    </html>
  )
}
