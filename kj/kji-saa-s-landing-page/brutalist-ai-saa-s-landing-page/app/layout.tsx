import type { Metadata, Viewport } from 'next'
import { JetBrains_Mono } from 'next/font/google'
import { GeistPixelGrid } from 'geist/font/pixel'
import { ThemeProvider } from '@/components/theme-provider'

import './globals.css'

const jetbrainsMono = JetBrains_Mono({
  subsets: ['latin'],
  variable: '--font-mono',
})

export const metadata: Metadata = {
  title: 'Brutalist AI SaaS Landing Page Template | Engineering-Grade Dark UI Kit',
  description:
    'A brutalist, engineering-themed landing page template for AI infrastructure and SaaS products. Features Geist Pixel typography, dot-grid backgrounds, live terminal animations, scramble-text micro-interactions, bento feature grids, isometric 3D illustrations, and a fully responsive dark industrial design system. Built with Next.js 16, Tailwind CSS, and Framer Motion.',
  keywords: [
    'brutalist landing page template',
    'AI SaaS template',
    'engineering UI kit',
    'Next.js landing page',
    'Tailwind CSS template',
    'dark UI template',
    'Geist Pixel font',
    'bento grid layout',
    'SaaS pricing page',
    'Framer Motion animations',
    'monospace design system',
    'developer landing page',
    'AI infrastructure template',
    'industrial web design',
    'dot matrix typography',
    'terminal UI components',
    'startup landing page',
    'tech landing page template',
  ],
  authors: [{ name: 'SYS.INT' }],
  creator: 'System Intelligence Corp.',
  publisher: 'System Intelligence Corp.',
  robots: {
    index: true,
    follow: true,
    googleBot: {
      index: true,
      follow: true,
      'max-video-preview': -1,
      'max-image-preview': 'large',
      'max-snippet': -1,
    },
  },
  openGraph: {
    type: 'website',
    locale: 'en_US',
    title: 'Brutalist AI SaaS Landing Page Template | Engineering-Grade Dark UI Kit',
    description:
      'A brutalist, engineering-themed landing page template for AI and SaaS products. Geist Pixel typography, terminal animations, bento grids, scramble-text effects, and a full industrial design system. Next.js 16 + Tailwind CSS + Framer Motion.',
    siteName: 'SYS.INT Template',
  },
  twitter: {
    card: 'summary_large_image',
    title: 'Brutalist AI SaaS Landing Page Template',
    description:
      'Engineering-grade brutalist template for AI SaaS products. Dot-grid backgrounds, live terminal animations, Geist Pixel typography, bento feature grids, and scramble-text micro-interactions. Built with Next.js 16.',
    creator: '@sysint',
  },
  category: 'technology',
}

export const viewport: Viewport = {
  themeColor: '#F2F1EA',
  width: 'device-width',
  initialScale: 1,
  maximumScale: 5,
}

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode
}>) {
  return (
    <html lang="en" className={`${jetbrainsMono.variable} ${GeistPixelGrid.variable}`} suppressHydrationWarning>
      <body className="font-mono antialiased">
        <ThemeProvider attribute="class" defaultTheme="light" enableSystem={false} disableTransitionOnChange>
          {children}
        </ThemeProvider>
      </body>
    </html>
  )
}
