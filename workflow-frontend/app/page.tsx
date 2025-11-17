// ============================================
// FILE: workflow-frontend/app/page.tsx
// ============================================
'use client';

import { useState, useCallback, useEffect } from 'react';
import WorkflowSubmit from './components/WorkflowSubmit';
import WorkflowStatus from './components/WorkflowStatus';
import SchedulerStats from './components/SchedulerStats';

export default function Home() {
    const [currentJobId, setCurrentJobId] = useState<string>('');
    const [userId, setUserId] = useState<string>('alice');
    const [completedJobs, setCompletedJobs] = useState<any[]>([]);
    const [schedulerStats, setSchedulerStats] = useState<any>(null);

    // Clear jobs when user changes (multi-user isolation)
    const handleUserIdChange = async (newUserId: string) => {
        if (newUserId !== userId && (currentJobId || completedJobs.length > 0)) {
            console.log(`Switching user from "${userId}" to "${newUserId}" - clearing session data`);
        }
        setUserId(newUserId);

        // Fetch user's active workflows BEFORE clearing current job
        let newWorkflowId = '';
        if (newUserId.trim()) {
            try {
                const response = await fetch('http://localhost:8000/workflows', {
                    headers: {
                        'X-User-ID': newUserId,
                    },
                });
                const data = await response.json();

                // Find the most recent RUNNING workflow
                const runningWorkflow = data.workflows?.find((w: any) => w.status === 'RUNNING');
                if (runningWorkflow) {
                    console.log(`Restoring active workflow: ${runningWorkflow.workflow_id}`);
                    newWorkflowId = runningWorkflow.workflow_id;
                }
            } catch (error) {
                console.error('Failed to fetch user workflows:', error);
            }
        }

        // Update state after fetch completes (prevents flash to 0%)
        setCurrentJobId(newWorkflowId);
        setCompletedJobs([]);
    };

    const handleJobCreated = useCallback((jobId: string) => {
        setCurrentJobId(jobId);
    }, []);

    const handleJobComplete = useCallback(() => {
        // Workflow completed - could fetch final results here
        setCurrentJobId('');
    }, []);

    // Fetch scheduler stats periodically
    useEffect(() => {
        const fetchStats = async () => {
            try {
                const response = await fetch('http://localhost:8000/scheduler/stats');
                const data = await response.json();
                setSchedulerStats(data);
            } catch (error) {
                console.error('Failed to fetch stats:', error);
            }
        };

        fetchStats();
        const interval = setInterval(fetchStats, 3000); // Update every 3s

        return () => clearInterval(interval);
    }, []);

    // Fetch active workflows on mount (for initial user)
    useEffect(() => {
        const fetchActiveWorkflows = async () => {
            if (!userId.trim()) return;

            try {
                const response = await fetch('http://localhost:8000/workflows', {
                    headers: {
                        'X-User-ID': userId,
                    },
                });
                const data = await response.json();

                // Find the most recent RUNNING workflow
                const runningWorkflow = data.workflows?.find((w: any) => w.status === 'RUNNING');
                if (runningWorkflow) {
                    console.log(`Found active workflow on load: ${runningWorkflow.workflow_id}`);
                    setCurrentJobId(runningWorkflow.workflow_id);
                }
            } catch (error) {
                console.error('Failed to fetch initial workflows:', error);
            }
        };

        fetchActiveWorkflows();
    }, []); // Run once on mount

    return (
        <main className="min-h-screen gradient-mesh bg-white">
            {/* Header with gradient */}
            <div className="gradient-purple text-white shadow-2xl relative overflow-hidden">
                {/* Animated background */}
                <div className="absolute inset-0 shimmer opacity-30"></div>
                <div className="max-w-7xl mx-auto px-8 py-10 relative z-10">
                    <div className="flex items-center gap-5">
                        <div className="w-16 h-16 bg-white/20 backdrop-blur-md rounded-2xl flex items-center justify-center flex-shrink-0 shadow-lg">
                            <svg className="w-9 h-9 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2m-3 7h3m-3 4h3m-6-4h.01M9 16h.01" />
                            </svg>
                        </div>
                        <div>
                            <h1 className="text-4xl font-bold tracking-tight drop-shadow-sm">
                                Branch-Aware Workflow Scheduler
                            </h1>
                            <p className="text-purple-100 mt-2 text-lg drop-shadow-sm">
                                ✨ DAG-based whole-slide image processing with smart tiling
                            </p>
                        </div>
                    </div>
                </div>
            </div>

            <div className="max-w-7xl mx-auto px-8 py-8">
                {/* User ID Selector - Fancy Glass Effect */}
                <div className="mb-8 glass rounded-3xl shadow-xl border border-purple-200 p-8 card-hover relative overflow-hidden">
                    <div className="absolute top-0 right-0 w-64 h-64 bg-gradient-to-br from-purple-200 to-pink-200 rounded-full blur-3xl opacity-20 -mr-32 -mt-32"></div>
                    <div className="flex items-center gap-6 relative z-10">
                        <div className="flex-shrink-0">
                            <div className="w-14 h-14 gradient-pink-purple rounded-2xl flex items-center justify-center flex-shrink-0 shadow-lg">
                                <svg className="w-7 h-7 text-white flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" />
                                </svg>
                            </div>
                        </div>
                        <div className="flex-1">
                            <label className="block text-sm font-bold text-gray-700 mb-2 tracking-wide">
                                User ID
                            </label>
                            <input
                                type="text"
                                value={userId}
                                onChange={(e) => handleUserIdChange(e.target.value)}
                                className="w-full border-2 border-purple-300 rounded-xl px-5 py-3 focus:border-purple-500 focus:ring-4 focus:ring-purple-100 outline-none transition-all shadow-sm font-medium"
                                placeholder="Enter user ID (e.g., alice, bob)"
                            />
                        </div>
                        <div className="flex-shrink-0">
                            <div className="gradient-blue-pink px-5 py-3 rounded-xl shadow-md">
                                <p className="text-sm text-white font-bold drop-shadow-sm">
                                    🔒 Max 3 Active Users
                                </p>
                            </div>
                        </div>
                    </div>
                </div>

                {/* Scheduler Stats - Show active workers, users, etc. */}
                {schedulerStats && (
                    <div className="mb-8">
                        <SchedulerStats stats={schedulerStats} />
                    </div>
                )}

                <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                    {/* Left Column - Workflow Submission */}
                    <div className="space-y-6">
                        <WorkflowSubmit
                            userId={userId}
                            onJobCreated={handleJobCreated}
                        />

                        {/* Quick Info Card - Fancy Gradient with Shine */}
                        <div className="gradient-purple-blue rounded-3xl shadow-2xl p-8 text-white relative overflow-hidden card-hover">
                            <div className="absolute inset-0 shimmer opacity-20"></div>
                            <div className="relative z-10">
                                <div className="flex items-center gap-3 mb-5">
                                    <div className="w-10 h-10 bg-white/20 backdrop-blur-sm rounded-xl flex items-center justify-center">
                                        <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                                        </svg>
                                    </div>
                                    <h3 className="text-2xl font-bold drop-shadow-md">
                                        How It Works
                                    </h3>
                                </div>
                                <ul className="space-y-4 text-base">
                                    <li className="flex items-start gap-3 bg-white/10 backdrop-blur-sm rounded-xl p-3">
                                        <span className="text-2xl">🌿</span>
                                        <span><strong className="font-bold">Branch-aware:</strong> Jobs in the same branch run sequentially, different branches run in parallel</span>
                                    </li>
                                    <li className="flex items-start gap-3 bg-white/10 backdrop-blur-sm rounded-xl p-3">
                                        <span className="text-2xl">⚡</span>
                                        <span><strong className="font-bold">Smart tiling:</strong> Skips 60-70% of background tiles for faster processing</span>
                                    </li>
                                    <li className="flex items-start gap-3 bg-white/10 backdrop-blur-sm rounded-xl p-3">
                                        <span className="text-2xl">👥</span>
                                        <span><strong className="font-bold">Multi-tenant:</strong> Only 3 users can be active at once, others queue</span>
                                    </li>
                                </ul>
                            </div>
                        </div>
                    </div>

                    {/* Right Column - Job Status */}
                    <div className="space-y-6">
                        {currentJobId ? (
                            <WorkflowStatus
                                workflowId={currentJobId}
                                userId={userId}
                                onComplete={handleJobComplete}
                            />
                        ) : (
                            <div className="bg-white rounded-2xl shadow-xl border border-purple-100 overflow-hidden">
                                <div className="gradient-purple px-6 py-4">
                                    <div className="flex items-center gap-3">
                                        <svg className="w-6 h-6 text-white flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2m-6 9l2 2 4-4" />
                                        </svg>
                                        <h2 className="text-xl font-bold text-white">Job Status</h2>
                                    </div>
                                </div>
                                <div className="p-6">
                                    <div className="text-center py-16">
                                        <div className="w-24 h-24 bg-purple-50 rounded-full flex items-center justify-center mx-auto mb-4 flex-shrink-0">
                                            <svg className="w-12 h-12 text-purple-300 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2" />
                                            </svg>
                                        </div>
                                        <p className="text-gray-400 font-medium">No active job</p>
                                        <p className="text-sm text-gray-300 mt-2">Upload images to get started</p>
                                    </div>
                                </div>
                            </div>
                        )}

                        {/* Completed Jobs */}
                        {completedJobs.length > 0 && (
                            <div className="bg-white rounded-2xl shadow-xl border border-purple-100 overflow-hidden">
                                <div className="bg-gradient-to-r from-green-500 to-green-600 px-6 py-4">
                                    <div className="flex items-center gap-3">
                                        <svg className="w-6 h-6 text-white flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                                        </svg>
                                        <h2 className="text-xl font-bold text-white">Completed Jobs</h2>
                                    </div>
                                </div>
                                <div className="p-6">
                                    <div className="space-y-3">
                                        {completedJobs.map((job, index) => (
                                            <div key={index} className="bg-green-50 border border-green-200 rounded-lg p-4">
                                                <div className="flex items-center justify-between">
                                                    <div>
                                                        <p className="font-semibold text-green-900">Job {index + 1}</p>
                                                        <p className="text-sm text-green-700">
                                                            {job.files?.length || 0} images processed
                                                        </p>
                                                    </div>
                                                    <div className="text-green-600">
                                                        <svg className="w-8 h-8" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                                                        </svg>
                                                    </div>
                                                </div>
                                            </div>
                                        ))}
                                    </div>
                                </div>
                            </div>
                        )}
                    </div>
                </div>
            </div>

            {/* Footer */}
            <div className="max-w-7xl mx-auto px-8 py-6 mt-12">
                <div className="text-center text-sm text-gray-500">
                    <p>Branch-Aware DAG Scheduler • Smart Tiling (60% speedup) • Multi-Tenant Isolation</p>
                </div>
            </div>
        </main>
    );
}
