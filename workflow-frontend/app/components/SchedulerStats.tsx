// ============================================
// FILE: workflow-frontend/app/components/SchedulerStats.tsx
// ============================================
'use client';

interface SchedulerStatsProps {
    stats: {
        active_workers: number;
        max_workers: number;
        active_users: number;
        max_active_users: number;
        queued_users: number;
        branches_active: number;
        total_branches: number;
        workflows_active: number;
    };
}

export default function SchedulerStats({ stats }: SchedulerStatsProps) {
    const workerUtilization = (stats.active_workers / stats.max_workers) * 100;
    const userUtilization = (stats.active_users / stats.max_active_users) * 100;

    return (
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            {/* Active Workers */}
            <div className="bg-white rounded-xl shadow-md border border-purple-100 p-4">
                <div className="flex items-center justify-between mb-2">
                    <h3 className="text-sm font-semibold text-gray-600">Active Workers</h3>
                    <svg className="w-5 h-5 text-purple-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                    </svg>
                </div>
                <div className="flex items-baseline gap-2">
                    <span className="text-3xl font-bold text-gray-900">{stats.active_workers}</span>
                    <span className="text-sm text-gray-500">/ {stats.max_workers}</span>
                </div>
                <div className="mt-2 w-full bg-gray-200 rounded-full h-2">
                    <div
                        className="bg-purple-600 h-2 rounded-full transition-all"
                        style={{ width: `${workerUtilization}%` }}
                    />
                </div>
                <p className="text-xs text-gray-500 mt-1">{workerUtilization.toFixed(0)}% utilized</p>
            </div>

            {/* Active Users */}
            <div className="bg-white rounded-xl shadow-md border border-blue-100 p-4">
                <div className="flex items-center justify-between mb-2">
                    <h3 className="text-sm font-semibold text-gray-600">Active Users</h3>
                    <svg className="w-5 h-5 text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4.354a4 4 0 110 5.292M15 21H3v-1a6 6 0 0112 0v1zm0 0h6v-1a6 6 0 00-9-5.197M13 7a4 4 0 11-8 0 4 4 0 018 0z" />
                    </svg>
                </div>
                <div className="flex items-baseline gap-2">
                    <span className="text-3xl font-bold text-gray-900">{stats.active_users}</span>
                    <span className="text-sm text-gray-500">/ {stats.max_active_users}</span>
                </div>
                <div className="mt-2 w-full bg-gray-200 rounded-full h-2">
                    <div
                        className="bg-blue-600 h-2 rounded-full transition-all"
                        style={{ width: `${userUtilization}%` }}
                    />
                </div>
                {stats.queued_users > 0 && (
                    <p className="text-xs text-orange-600 font-medium mt-1">
                        {stats.queued_users} waiting in queue
                    </p>
                )}
            </div>

            {/* Active Branches */}
            <div className="bg-white rounded-xl shadow-md border border-green-100 p-4">
                <div className="flex items-center justify-between mb-2">
                    <h3 className="text-sm font-semibold text-gray-600">Active Branches</h3>
                    <svg className="w-5 h-5 text-green-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 20l4-16m2 16l4-16M6 9h14M4 15h14" />
                    </svg>
                </div>
                <div className="flex items-baseline gap-2">
                    <span className="text-3xl font-bold text-gray-900">{stats.branches_active}</span>
                    <span className="text-sm text-gray-500">/ {stats.total_branches}</span>
                </div>
                <p className="text-xs text-gray-500 mt-3">
                    {stats.branches_active === 0 ? 'All idle' : 'Processing jobs'}
                </p>
            </div>

            {/* Active Workflows */}
            <div className="bg-white rounded-xl shadow-md border border-orange-100 p-4">
                <div className="flex items-center justify-between mb-2">
                    <h3 className="text-sm font-semibold text-gray-600">Active Workflows</h3>
                    <svg className="w-5 h-5 text-orange-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2" />
                    </svg>
                </div>
                <div className="flex items-baseline gap-2">
                    <span className="text-3xl font-bold text-gray-900">{stats.workflows_active}</span>
                </div>
                <p className="text-xs text-gray-500 mt-3">
                    {stats.workflows_active === 0 ? 'No active workflows' : 'In progress'}
                </p>
            </div>
        </div>
    );
}
