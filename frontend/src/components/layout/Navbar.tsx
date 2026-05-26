'use client';

import { usePathname } from 'next/navigation';
import { Activity } from 'lucide-react';
import { useAuth } from '@/contexts/AuthContext';
import NotificationBell from '@/components/notifications/NotificationBell';

const routeLabels: Record<string, string> = {
  '/dashboard': 'Dashboard',
  '/upload': 'Upload Document',
  '/documents': 'My Documents',
  '/chat': 'Conversational AI Chat',
};

function getRouteLabel(pathname: string): string {
  if (pathname.startsWith('/documents/') && pathname !== '/documents') {
    return 'Document Details';
  }
  return routeLabels[pathname] || 'HealthAI';
}

export default function Navbar() {
  const pathname = usePathname();
  const { user } = useAuth();

  return (
    <header className="flex items-center justify-between px-6 py-4 border-b"
      style={{ borderColor: 'rgba(59,130,246,0.1)', background: 'rgba(10,15,30,0.6)', backdropFilter: 'blur(20px)' }}>
      {/* Page Title */}
      <div className="flex items-center gap-3">
        {/* Mobile logo */}
        <div className="w-8 h-8 rounded-lg flex items-center justify-center md:hidden"
          style={{ background: 'linear-gradient(135deg, #3b82f6, #8b5cf6)' }}>
          <Activity className="w-4 h-4 text-white" />
        </div>
        <div>
          <h2 className="font-semibold text-white">{getRouteLabel(pathname)}</h2>
          <p className="text-xs text-slate-500 hidden sm:block">Healthcare Insurance Document Intelligence</p>
        </div>
      </div>

      {/* Right side */}
      <div className="flex items-center gap-3">
        {/* Notification Bell */}
        <NotificationBell />

        {/* Avatar */}
        <div className="w-9 h-9 rounded-xl flex items-center justify-center text-sm font-bold text-white"
          style={{ background: 'linear-gradient(135deg, #3b82f6, #8b5cf6)' }}>
          {user?.full_name?.[0]?.toUpperCase() || 'U'}
        </div>
      </div>
    </header>
  );
}
