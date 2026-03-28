import UploadCard from "../components/UploadCard";

export default function HomePage() {
  return (
    <main className="min-h-screen bg-slate-50 p-8">
      <div className="mx-auto max-w-4xl space-y-6">
        <div>
          <h1 className="text-3xl font-bold">Autonomous Decision Intelligence</h1>
          <p className="mt-2 text-gray-600">
            Upload a tabular dataset and run the multi-step agent workflow.
          </p>
        </div>

        <UploadCard />
      </div>
    </main>
  );
}