'use client';

import { useState, useEffect, useRef } from 'react';
import axios from 'axios';

interface WorkflowStatusProps {
    workflowId: string;
    userId: string;
    onComplete: () => void;
}

interface Job {
    job_id: string;
    image: string;
    type: string;
    status: string;
    result?: {
        num_cells?: number;
        processing_time?: number;
        results?: {
            mask: string;
            overlay: string;
            result: string;
            original: string;
        };
        message?: string;
    };
}

interface Branch {
    branch_id: string;
    jobs: Job[];
    status: string;
    active_job?: string;
}

interface WorkflowData {
    workflow_id: string;
    name: string;
    status: string;
    progress: number;
    branches: Branch[];
    total_jobs: number;
    completed_jobs: number;
    running_jobs: number;
    queued_jobs: number;
}

export default function WorkflowStatus({ workflowId, userId, onComplete }: WorkflowStatusProps) {
    const [workflow, setWorkflow] = useState<WorkflowData | null>(null);
    const [error, setError] = useState<string>('');
    const timeoutRef = useRef<NodeJS.Timeout | null>(null);

    useEffect(() => {
        setWorkflow(null);
        setError('');

        let isMounted = true;

        const pollStatus = async () => {
            if (!isMounted) return;

            try {
                const response = await axios.get(
                    `http://localhost:8000/workflows/${workflowId}`,
                    {
                        headers: {
                            'X-User-ID': userId,
                        },
                    }
                );

                if (!isMounted) return;

                const workflowData = response.data;
                setWorkflow(workflowData);

                // Stop polling if workflow is in a terminal state
                if (['COMPLETED', 'FAILED', 'CANCELLED'].includes(workflowData.status)) {
                    if (workflowData.status === 'COMPLETED') {
                        onComplete();
                    }
                } else {
                    // Continue polling
                    timeoutRef.current = setTimeout(pollStatus, 2000);
                }
            } catch (err: any) {
                if (!isMounted) return;

                console.error('Workflow status polling error:', err);
                if (err.response?.status === 404) {
                    setError('Workflow not found or access denied');
                } else {
                    setError(err.response?.data?.detail || 'Failed to fetch workflow status');
                }
            }
        };

        pollStatus();

        return () => {
            isMounted = false;
            if (timeoutRef.current) {
                clearTimeout(timeoutRef.current);
            }
        };
    }, [workflowId, userId, onComplete]);

    const getStatusColor = (status: string) => {
        switch (status) {
            case 'COMPLETED':
                return 'bg-green-100 text-green-800 border-green-200';
            case 'FAILED':
                return 'bg-red-100 text-red-800 border-red-200';
            case 'CANCELLED':
                return 'bg-orange-100 text-orange-800 border-orange-200';
            case 'RUNNING':
                return 'bg-blue-100 text-blue-800 border-blue-200';
            case 'QUEUED':
                return 'bg-yellow-100 text-yellow-800 border-yellow-200';
            default:
                return 'bg-gray-100 text-gray-800 border-gray-200';
        }
    };

    const getJobStatusIcon = (status: string) => {
        switch (status) {
            case 'COMPLETED':
                return (
                    <svg className="w-4 h-4 text-green-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                    </svg>
                );
            case 'RUNNING':
                return (
                    <svg className="w-4 h-4 text-blue-600 animate-spin" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                    </svg>
                );
            case 'FAILED':
                return (
                    <svg className="w-4 h-4 text-red-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                    </svg>
                );
            case 'QUEUED':
                return (
                    <svg className="w-4 h-4 text-yellow-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
                    </svg>
                );
            default:
                return null;
        }
    };

    const getImageUrl = (filePath: string) => {
        // Extract workflow_name and filename from path
        // e.g., "outputs/alice/My_Workflow/job1_mask.png" -> workflow_name="My_Workflow", filename="job1_mask.png"
        const parts = filePath.split('/');
        if (parts.length >= 4) {
            const workflow_name = parts[2];  // outputs/alice/[workflow_name]/filename
            const filename = parts[3];
            return `http://localhost:8000/outputs/${userId}/${workflow_name}/${filename}`;
        }
        // Fallback for old format (shouldn't happen with new code)
        const filename = filePath.split('/').pop();
        return `http://localhost:8000/outputs/${userId}/unknown/${filename}`;
    };

    // Get all completed jobs with results
    const completedJobsWithResults = workflow?.branches
        .flatMap(branch => branch.jobs)
        .filter(job => job.status === 'COMPLETED' && job.result?.results) || [];

    if (error) {
        return (
            <div className="bg-white p-6 rounded-lg shadow">
                <div className="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded">
                    {error}
                </div>
            </div>
        );
    }

    if (!workflow) {
        return (
            <div className="bg-white p-6 rounded-lg shadow">
                <div className="flex items-center justify-center py-8">
                    <svg className="animate-spin h-8 w-8 text-blue-600" fill="none" viewBox="0 0 24 24">
                        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                    </svg>
                    <span className="ml-3 text-gray-600">Loading workflow...</span>
                </div>
            </div>
        );
    }

    return (
        <div className="glass p-8 rounded-3xl shadow-xl border border-purple-200 card-hover">
            <h2 className="text-2xl font-bold mb-6 bg-gradient-to-r from-purple-600 to-blue-600 bg-clip-text text-transparent">
                Workflow Progress
            </h2>

            <div className="space-y-4">
                {/* Workflow Info */}
                <div>
                    <p className="text-sm text-gray-600">Workflow Name:</p>
                    <p className="font-medium text-gray-900">{workflow.name}</p>
                </div>

                <div>
                    <p className="text-sm text-gray-600">Workflow ID:</p>
                    <p className="text-xs font-mono bg-gray-50 p-2 rounded break-all">
                        {workflow.workflow_id}
                    </p>
                </div>

                {/* Status */}
                <div className={`px-4 py-3 rounded border ${getStatusColor(workflow.status)}`}>
                    <p className="font-medium">Status: {workflow.status}</p>
                    <p className="text-sm mt-1">Progress: {workflow.progress}%</p>
                </div>

                {/* Progress Bar - Fancy Gradient */}
                <div className="w-full bg-gray-200 rounded-full h-4 shadow-inner overflow-hidden">
                    <div
                        className="gradient-pink-purple h-4 rounded-full transition-all duration-500 shadow-lg relative shimmer"
                        style={{ width: `${workflow.progress}%` }}
                    />
                </div>

                {/* Job Statistics */}
                <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                    <div className="bg-gray-50 p-3 rounded">
                        <p className="text-xs text-gray-600">Total Jobs</p>
                        <p className="text-2xl font-bold text-gray-900">{workflow.total_jobs}</p>
                    </div>
                    <div className="bg-green-50 p-3 rounded">
                        <p className="text-xs text-green-600">Completed</p>
                        <p className="text-2xl font-bold text-green-900">{workflow.completed_jobs}</p>
                    </div>
                    <div className="bg-blue-50 p-3 rounded">
                        <p className="text-xs text-blue-600">Running</p>
                        <p className="text-2xl font-bold text-blue-900">{workflow.running_jobs}</p>
                    </div>
                    <div className="bg-yellow-50 p-3 rounded">
                        <p className="text-xs text-yellow-600">Queued</p>
                        <p className="text-2xl font-bold text-yellow-900">{workflow.queued_jobs}</p>
                    </div>
                </div>

                {/* Branches */}
                <div>
                    <h3 className="text-lg font-bold text-gray-900 mb-4">Branches</h3>
                    <div className="space-y-4">
                        {workflow.branches.map((branch) => (
                            <div key={branch.branch_id} className="border-2 border-purple-200 bg-purple-50/30 rounded-2xl p-5 card-hover">
                                <div className="flex items-center justify-between mb-3">
                                    <h4 className="font-medium text-gray-900">{branch.branch_id}</h4>
                                    <span className={`px-2 py-1 text-xs rounded ${getStatusColor(branch.status)}`}>
                                        {branch.status}
                                    </span>
                                </div>

                                {/* Jobs in this branch */}
                                <div className="space-y-2">
                                    {branch.jobs.map((job) => (
                                        <div
                                            key={job.job_id}
                                            className={`flex items-center gap-3 p-2 rounded ${
                                                job.status === 'RUNNING' ? 'bg-blue-50' : 'bg-gray-50'
                                            }`}
                                        >
                                            <div className="flex-shrink-0">
                                                {getJobStatusIcon(job.status)}
                                            </div>
                                            <div className="flex-1 min-w-0">
                                                <p className="text-sm font-medium text-gray-900 truncate">
                                                    {job.job_id}
                                                </p>
                                                <p className="text-xs text-gray-600 truncate">
                                                    {job.image}
                                                </p>
                                            </div>
                                            <div className="flex-shrink-0">
                                                <span className="text-xs text-gray-500">{job.status}</span>
                                            </div>
                                        </div>
                                    ))}
                                </div>
                            </div>
                        ))}
                    </div>
                </div>

                {/* Results Gallery */}
                {completedJobsWithResults.length > 0 && (
                    <div className="mt-6">
                        <h3 className="text-lg font-medium text-gray-900 mb-3 flex items-center gap-2">
                            <svg className="w-5 h-5 text-green-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
                            </svg>
                            Segmentation Results
                        </h3>
                        <div className="space-y-6">
                            {completedJobsWithResults.map((job) => (
                                <div key={job.job_id} className="border-2 border-green-200 rounded-lg p-4 bg-green-50">
                                    {/* Job Header */}
                                    <div className="mb-4">
                                        <div className="flex items-center justify-between mb-2">
                                            <h4 className="font-semibold text-gray-900">{job.job_id}</h4>
                                            <span className="px-2 py-1 bg-green-600 text-white text-xs rounded-full">
                                                ✓ Completed
                                            </span>
                                        </div>
                                        <p className="text-sm text-gray-600">{job.result?.message}</p>

                                        {/* Metadata */}
                                        <div className="flex gap-4 mt-2">
                                            {job.result?.num_cells !== undefined && (
                                                <div className="flex items-center gap-1 text-sm">
                                                    <svg className="w-4 h-4 text-purple-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                                                    </svg>
                                                    <span className="font-semibold text-purple-900">
                                                        {job.result.num_cells} cells detected
                                                    </span>
                                                </div>
                                            )}
                                            {job.result?.processing_time !== undefined && (
                                                <div className="flex items-center gap-1 text-sm">
                                                    <svg className="w-4 h-4 text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
                                                    </svg>
                                                    <span className="text-blue-900">
                                                        {job.result.processing_time.toFixed(2)}s
                                                    </span>
                                                </div>
                                            )}
                                        </div>
                                    </div>

                                    {/* Image Gallery */}
                                    {job.result?.results && (
                                        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                                            {/* Segmentation Mask */}
                                            <div className="bg-white rounded-lg overflow-hidden shadow">
                                                <div className="bg-gray-800 text-white px-3 py-2 text-sm font-medium">
                                                    Segmentation Mask
                                                </div>
                                                <img
                                                    src={getImageUrl(job.result.results.mask)}
                                                    alt="Segmentation Mask"
                                                    className="w-full h-48 object-cover"
                                                />
                                            </div>

                                            {/* Colored Overlay */}
                                            <div className="bg-white rounded-lg overflow-hidden shadow">
                                                <div className="bg-purple-600 text-white px-3 py-2 text-sm font-medium">
                                                    Colored Overlay
                                                </div>
                                                <img
                                                    src={getImageUrl(job.result.results.overlay)}
                                                    alt="Colored Overlay"
                                                    className="w-full h-48 object-cover"
                                                />
                                            </div>

                                            {/* Annotated Result */}
                                            <div className="bg-white rounded-lg overflow-hidden shadow">
                                                <div className="bg-green-600 text-white px-3 py-2 text-sm font-medium">
                                                    Annotated Result
                                                </div>
                                                <img
                                                    src={getImageUrl(job.result.results.result)}
                                                    alt="Annotated Result"
                                                    className="w-full h-48 object-cover"
                                                />
                                            </div>
                                        </div>
                                    )}
                                </div>
                            ))}
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
}
