'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import {
  Activity, LayoutDashboard, FileText, Upload, Shield, Settings, LogOut, Scale,
} from 'lucide-react';
import { useAuth } from '@/contexts/AuthContext';

const navItems = [
  { href: '/dashboard', label: 'Dashboard', icon: LayoutDashboard },
  { href: '/upload', label: 'Upload', icon: Upload },
  { href: '/documents', label: 'Documents', icon: FileText },
  { href: '/compare', label: 'Compare Policies', icon: Scale },
];

export default function Sidebar() {
  const pathname = usePathname();
  const { user, logout } = useAuth();

  return (
    <aside className="sidebar w-60 flex-shrink-0 flex flex-col h-full hidden md:flex">
      {/* Logo */}
      <div className="flex items-center gap-3 px-5 py-5 border-b border-slate-700/40">
        <div className="w-9 h-9 rounded-xl flex items-center justify-center"
          style={{ background: 'linear-gradient(135deg, #3b82f6, #8b5cf6)' }}>
          <Activity className="w-5 h-5 text-white" />
        </div>
        <div>
          <p className="font-bold text-white text-sm">HealthAI</p>
          <p className="text-xs text-slate-500">Insurance Intel</p>
        </div>
      </div>

      {/* Nav */}
      <nav className="flex-1 px-3 py-4 space-y-1">
        <p className="px-3 text-xs font-semibold text-slate-600 uppercase tracking-wider mb-3">Menu</p>
        {navItems.map(({ href, label, icon: Icon }) => {
          const isActive = pathname === href || pathname.startsWith(`${href}/`);
          return (
            <Link
              key={href}
              href={href}
              className={`flex items-center gap-3 px-3 py-2.5 rounded-xl text-sm font-medium transition-all ${
                isActive
                  ? 'text-white'
                  : 'text-slate-400 hover:text-white hover:bg-white/5'
              }`}
              style={isActive ? { background: 'linear-gradient(135deg, rgba(59,130,246,0.2), rgba(139,92,246,0.15))', border: '1px solid rgba(59,130,246,0.3)' } : {}}
            >
              <Icon className="w-4 h-4" />
              {label}
            </Link>
          );
        })}
      </nav>

      {/* User + Logout */}
      <div className="px-3 py-4 border-t border-slate-700/40 space-y-1">
        <div className="px-3 py-3 rounded-xl" style={{ background: 'rgba(59,130,246,0.05)' }}>
          <p className="text-sm font-semibold text-white truncate">{user?.full_name}</p>
          <p className="text-xs text-slate-500 truncate">{user?.email}</p>
          <span className="inline-flex mt-1 px-2 py-0.5 text-xs rounded-full capitalize"
            style={{ background: 'rgba(16,185,129,0.15)', color: '#34d399' }}>
            {user?.role}
          </span>
        </div>
        <button
          onClick={logout}
          id="logout-btn"
          className="w-full flex items-center gap-3 px-3 py-2.5 rounded-xl text-sm text-slate-400 hover:text-red-400 hover:bg-red-400/5 transition-all"
        >
          <LogOut className="w-4 h-4" />
          Sign Out
        </button>
      </div>
    </aside>
  );
}
