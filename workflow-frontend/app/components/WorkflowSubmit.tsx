'use client';

import { useState } from 'react';
import axios from 'axios';

interface WorkflowSubmitProps {
    userId: string;
    onJobCreated: (workflowId: string) => void;
}

interface Job {
    job_id: string;
    image: string;
    type: string;
    file?: File;
}

interface Branch {
    id: string;
    name: string;
    jobs: Job[];
}

export default function WorkflowSubmit({ userId, onJobCreated }: WorkflowSubmitProps) {
    const [workflowName, setWorkflowName] = useState('');
    const [branches, setBranches] = useState<Branch[]>([
        { id: 'branch_A', name: 'Branch A', jobs: [] }
    ]);
    const [uploading, setUploading] = useState(false);
    const [error, setError] = useState<string>('');

    const addBranch = () => {
        const branchLetter = String.fromCharCode(65 + branches.length); // A, B, C, ...
        setBranches([...branches, {
            id: `branch_${branchLetter}`,
            name: `Branch ${branchLetter}`,
            jobs: []
        }]);
    };

    const removeBranch = (branchId: string) => {
        if (branches.length <= 1) {
            setError('Must have at least one branch');
            return;
        }
        setBranches(branches.filter(b => b.id !== branchId));
    };

    const addJob = (branchId: string) => {
        setBranches(branches.map(branch => {
            if (branch.id === branchId) {
                const jobNumber = branch.jobs.length + 1;
                return {
                    ...branch,
                    jobs: [...branch.jobs, {
                        job_id: `${branchId}_job${jobNumber}`,
                        image: '',
                        type: 'segment'
                    }]
                };
            }
            return branch;
        }));
    };

    const removeJob = (branchId: string, jobId: string) => {
        setBranches(branches.map(branch => {
            if (branch.id === branchId) {
                return {
                    ...branch,
                    jobs: branch.jobs.filter(j => j.job_id !== jobId)
                };
            }
            return branch;
        }));
    };

    const handleFileUpload = async (branchId: string, jobId: string, file: File) => {
        if (!file) return;

        // Upload file to backend
        try {
            const formData = new FormData();
            formData.append('file', file);

            const response = await axios.post(
                `http://localhost:8000/upload/${userId}`,
                formData,
                {
                    headers: {
                        'Content-Type': 'multipart/form-data',
                    },
                }
            );

            const uploadedPath = response.data.filepath;

            // Update job with uploaded file path
            setBranches(branches.map(branch => {
                if (branch.id === branchId) {
                    return {
                        ...branch,
                        jobs: branch.jobs.map(job =>
                            job.job_id === jobId ? { ...job, image: uploadedPath, file } : job
                        )
                    };
                }
                return branch;
            }));
        } catch (err: any) {
            console.error('File upload error:', err);
            setError(`Failed to upload ${file.name}: ${err.response?.data?.detail || err.message}`);
        }
    };

    const handleSubmit = async () => {
        setError('');

        // Validation
        if (!workflowName.trim()) {
            setError('Please enter a workflow name');
            return;
        }

        const totalJobs = branches.reduce((sum, b) => sum + b.jobs.length, 0);
        if (totalJobs === 0) {
            setError('Please add at least one job');
            return;
        }

        // Check all jobs have files uploaded
        for (const branch of branches) {
            for (const job of branch.jobs) {
                if (!job.image.trim()) {
                    setError(`Please upload a file for ${job.job_id}`);
                    return;
                }
            }
        }

        setUploading(true);

        try {
            // Build DAG structure
            const dag: Record<string, Job[]> = {};
            branches.forEach(branch => {
                if (branch.jobs.length > 0) {
                    dag[branch.id] = branch.jobs;
                }
            });

            const workflow = {
                name: workflowName,
                dag
            };

            console.log('Submitting workflow:', workflow);

            // Submit to backend
            const response = await axios.post(
                'http://localhost:8000/workflows',
                workflow,
                {
                    headers: {
                        'X-User-ID': userId,
                    },
                }
            );

            console.log('Workflow response:', response.data);

            // Check if workflow was queued or accepted
            if (response.data.status === 'QUEUED') {
                setError(`Queued: ${response.data.message}`);
            } else {
                // Notify parent component
                onJobCreated(response.data.workflow_id);

                // Reset form
                setWorkflowName('');
                setBranches([{ id: 'branch_A', name: 'Branch A', jobs: [] }]);
            }

        } catch (err: any) {
            console.error('Submit error:', err);
            setError(err.response?.data?.detail || 'Submission failed. Please try again.');
        } finally {
            setUploading(false);
        }
    };

    return (
        <div className="glass p-8 rounded-3xl shadow-xl border border-purple-200 card-hover">
            <h2 className="text-2xl font-bold mb-6 bg-gradient-to-r from-purple-600 to-pink-600 bg-clip-text text-transparent">
                Create Workflow
            </h2>

            <div className="space-y-4">
                {/* Workflow Name */}
                <div>
                    <label className="block text-sm font-bold text-gray-700 mb-2 tracking-wide">
                        Workflow Name
                    </label>
                    <input
                        type="text"
                        value={workflowName}
                        onChange={(e) => setWorkflowName(e.target.value)}
                        placeholder="e.g., Cell Segmentation Pipeline"
                        className="w-full border-2 border-purple-300 rounded-xl px-5 py-3 focus:border-purple-500 focus:ring-4 focus:ring-purple-100 outline-none transition-all shadow-sm font-medium"
                    />
                </div>

                {/* Branches */}
                {branches.map((branch, branchIndex) => (
                    <div key={branch.id} className="border-2 border-purple-200 bg-purple-50/50 rounded-2xl p-5 card-hover">
                        <div className="flex justify-between items-center mb-4">
                            <h3 className="font-bold text-purple-900 text-lg flex items-center gap-2">
                                <span className="w-2 h-2 bg-purple-500 rounded-full"></span>
                                {branch.name}
                            </h3>
                            <button
                                onClick={() => removeBranch(branch.id)}
                                disabled={branches.length <= 1}
                                className="px-3 py-1.5 text-sm font-medium rounded-lg text-red-600 hover:bg-red-50 disabled:text-gray-400 disabled:hover:bg-transparent transition-colors"
                            >
                                Remove
                            </button>
                        </div>

                        {/* Jobs in this branch */}
                        <div className="space-y-2 mb-3">
                            {branch.jobs.map((job, jobIndex) => (
                                <div key={job.job_id} className="flex gap-2 items-center">
                                    <span className="text-sm text-gray-600 w-20">Job {jobIndex + 1}:</span>
                                    <div className="flex-1 flex items-center gap-2">
                                        <label className="cursor-pointer flex-1">
                                            <input
                                                type="file"
                                                accept=".svs,.tif,.tiff"
                                                onChange={(e) => {
                                                    const file = e.target.files?.[0];
                                                    if (file) {
                                                        handleFileUpload(branch.id, job.job_id, file);
                                                    }
                                                }}
                                                className="hidden"
                                            />
                                            <div className={`border-2 border-dashed rounded px-3 py-2 text-sm text-center transition-colors ${
                                                job.image
                                                    ? 'border-green-300 bg-green-50 text-green-700'
                                                    : 'border-gray-300 bg-gray-50 text-gray-500 hover:border-blue-400 hover:bg-blue-50'
                                            }`}>
                                                {job.image ? (
                                                    <div className="flex items-center justify-center gap-2">
                                                        <svg className="w-4 h-4 text-green-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                                                        </svg>
                                                        <span className="truncate">{job.file?.name || job.image}</span>
                                                    </div>
                                                ) : (
                                                    '📁 Click to upload .svs file'
                                                )}
                                            </div>
                                        </label>
                                    </div>
                                    <button
                                        onClick={() => removeJob(branch.id, job.job_id)}
                                        className="text-red-600 hover:text-red-700 text-sm px-2"
                                    >
                                        ✕
                                    </button>
                                </div>
                            ))}
                        </div>

                        <button
                            onClick={() => addJob(branch.id)}
                            className="px-4 py-2 gradient-purple-blue text-white rounded-xl text-sm font-bold hover:shadow-lg transition-all"
                        >
                            + Add Job
                        </button>
                    </div>
                ))}

                {/* Add Branch Button */}
                <button
                    onClick={addBranch}
                    className="w-full border-2 border-dashed border-purple-300 bg-purple-50/30 rounded-2xl py-4 text-purple-700 hover:border-purple-500 hover:bg-purple-100 font-bold transition-all hover:shadow-md"
                >
                    ✨ Add Branch (Parallel Execution)
                </button>

                {/* Submit Button */}
                <button
                    onClick={handleSubmit}
                    disabled={uploading}
                    className="w-full gradient-pink-purple text-white py-4 px-6 rounded-2xl font-bold text-lg hover:shadow-2xl disabled:opacity-50 disabled:cursor-not-allowed transition-all transform hover:scale-105 disabled:transform-none"
                >
                    {uploading ? '⏳ Submitting...' : '🚀 Submit Workflow'}
                </button>

                {/* Error Message */}
                {error && (
                    <div className="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded">
                        {error}
                    </div>
                )}
            </div>
        </div>
    );
}
