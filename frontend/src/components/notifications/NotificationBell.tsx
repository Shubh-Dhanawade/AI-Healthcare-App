'use client';

import { useState, useEffect, useRef } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { remindersApi } from '@/lib/apiHelpers';
import { Bell, Calendar, DollarSign, X, Check, ArrowRight, Loader2 } from 'lucide-react';
import Link from 'next/link';
import toast from 'react-hot-toast';

interface Reminder {
  id: string;
  document_id: string;
  title: string;
  reminder_type: 'renewal' | 'premium';
  reminder_date: string;
  premium_amount?: number;
  is_dismissed: boolean;
}

export default function NotificationBell() {
  const [isOpen, setIsOpen] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);
  const queryClient = useQueryClient();

  // Fetch reminders
  const { data: reminders = [], isLoading } = useQuery<Reminder[]>({
    queryKey: ['reminders'],
    queryFn: remindersApi.list,
    refetchInterval: 30000, // Poll every 30 seconds
  });

  // Close dropdown on click outside
  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      if (containerRef.current && !containerRef.current.contains(event.target as Node)) {
        setIsOpen(false);
      }
    }
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  // Mutation to dismiss a reminder
  const dismissMutation = useMutation({
    mutationFn: remindersApi.dismiss,
    onSuccess: (_, variables) => {
      queryClient.setQueryData(['reminders'], (prev: Reminder[] | undefined) => 
        prev ? prev.filter(r => r.id !== variables) : []
      );
      toast.success("Reminder dismissed");
    },
    onError: () => {
      toast.error("Failed to dismiss reminder");
    }
  });

  const activeReminders = reminders.filter(r => !r.is_dismissed);
  const count = activeReminders.length;

  return (
    <div className="relative" ref={containerRef}>
      {/* Bell Button */}
      <button
        onClick={() => setIsOpen(!isOpen)}
        id="notification-bell-btn"
        className="relative w-9 h-9 rounded-xl flex items-center justify-center transition-all hover:bg-white/5"
        style={{ border: '1px solid rgba(59,130,246,0.2)' }}
      >
        <Bell className={`w-4 h-4 transition-all ${count > 0 ? 'text-amber-400 animate-pulse' : 'text-slate-400'}`} />
        {count > 0 && (
          <span className="absolute -top-1 -right-1 w-4 h-4 bg-amber-500 rounded-full flex items-center justify-center text-[10px] font-black text-black">
            {count}
          </span>
        )}
      </button>

      {/* Dropdown Card */}
      {isOpen && (
        <div className="absolute right-0 mt-3 w-80 sm:w-96 bg-[#111827] border border-slate-700/60 rounded-2xl shadow-2xl z-50 overflow-hidden backdrop-blur-xl">
          {/* Header */}
          <div className="px-4 py-3.5 border-b border-slate-800 flex items-center justify-between">
            <h3 className="text-sm font-bold text-white flex items-center gap-2">
              Notifications
              {count > 0 && (
                <span className="px-2 py-0.5 text-xs bg-amber-500/10 text-amber-400 border border-amber-500/20 rounded-full">
                  {count} Active
                </span>
              )}
            </h3>
            {count > 0 && (
              <p className="text-[10px] text-slate-500 uppercase tracking-wider font-semibold">
                Alerts triggered
              </p>
            )}
          </div>

          {/* Body */}
          <div className="max-h-80 overflow-y-auto divide-y divide-slate-800/60 scrollbar">
            {isLoading ? (
              <div className="flex items-center justify-center py-8 gap-2 text-sm text-slate-500">
                <Loader2 className="w-4 h-4 animate-spin text-blue-500" />
                Loading alerts...
              </div>
            ) : activeReminders.length === 0 ? (
              <div className="py-10 text-center">
                <Bell className="w-8 h-8 text-slate-600 mx-auto mb-2" />
                <p className="text-slate-400 text-sm">All caught up!</p>
                <p className="text-xs text-slate-600 mt-1">No active reminders or deadlines.</p>
              </div>
            ) : (
              activeReminders.map((reminder) => {
                const isRenewal = reminder.reminder_type === 'renewal';
                const date = new Date(reminder.reminder_date);
                
                return (
                  <div key={reminder.id} className="p-4 hover:bg-white/2 transition-all flex items-start gap-3">
                    {/* Icon Column */}
                    <div className={`w-8 h-8 rounded-lg flex items-center justify-center flex-shrink-0 mt-0.5 ${
                      isRenewal 
                        ? 'bg-amber-500/10 text-amber-400 border border-amber-500/20' 
                        : 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/20'
                    }`}>
                      {isRenewal ? <Calendar className="w-4 h-4" /> : <DollarSign className="w-4 h-4" />}
                    </div>

                    {/* Content Column */}
                    <div className="flex-1 min-w-0">
                      <p className="text-xs font-semibold text-slate-200 leading-tight">
                        {reminder.title}
                      </p>
                      
                      {reminder.premium_amount && (
                        <p className="text-xs text-slate-400 font-medium mt-1">
                          Amount due: <span className="text-emerald-400 font-bold">${reminder.premium_amount}</span>
                        </p>
                      )}

                      <p className="text-[10px] text-slate-500 mt-1.5 flex items-center gap-1 font-medium">
                        Alert date: {date.toLocaleDateString(undefined, { month: 'short', day: 'numeric' })}
                      </p>
                      
                      {/* Nav Link */}
                      <Link 
                        href={`/documents/${reminder.document_id}`}
                        onClick={() => setIsOpen(false)}
                        className="inline-flex items-center gap-1 text-[10px] font-bold text-blue-400 hover:text-blue-300 mt-2 transition-all"
                      >
                        View Policy <ArrowRight className="w-3 h-3" />
                      </Link>
                    </div>

                    {/* Actions Column */}
                    <button
                      onClick={() => dismissMutation.mutate(reminder.id)}
                      disabled={dismissMutation.isPending}
                      className="p-1 rounded bg-slate-800 hover:bg-slate-700 text-slate-400 hover:text-white transition-all flex-shrink-0 self-start"
                      title="Dismiss alert"
                    >
                      <X className="w-3.5 h-3.5" />
                    </button>
                  </div>
                );
              })
            )}
          </div>
        </div>
      )}
    </div>
  );
}
