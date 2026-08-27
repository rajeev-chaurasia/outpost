import { useEffect, useState } from 'react'
import type { AuditEntry, QueryResponse, TenantSummary } from './api/client'
import { listAudit, listTenants, queryTenant } from './api/client'
import { AnswerCard } from './components/AnswerCard'
import { AuditTable } from './components/AuditTable'
import { QueryPanel } from './components/QueryPanel'
import { TenantPicker } from './components/TenantPicker'
import './App.css'

function App() {
  const [tenants, setTenants] = useState<TenantSummary[]>([])
  const [selectedTenant, setSelectedTenant] = useState<string | null>(null)
  const [result, setResult] = useState<QueryResponse | null>(null)
  const [auditEntries, setAuditEntries] = useState<AuditEntry[]>([])
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    listTenants()
      .then((fetched) => {
        setTenants(fetched)
        if (fetched.length > 0) setSelectedTenant(fetched[0].tenant_id)
      })
      .catch((err: unknown) => setError(String(err)))
  }, [])

  useEffect(() => {
    if (!selectedTenant) return
    setResult(null)
    listAudit(selectedTenant)
      .then(setAuditEntries)
      .catch((err: unknown) => setError(String(err)))
  }, [selectedTenant])

  async function handleAsk(question: string) {
    if (!selectedTenant) return
    setIsLoading(true)
    setError(null)
    try {
      const response = await queryTenant(selectedTenant, question)
      setResult(response)
      setAuditEntries(await listAudit(selectedTenant))
    } catch (err) {
      setError(String(err))
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <div className="app">
      <header>
        <h1>outpost</h1>
        <p className="tagline">Ask a tenant-scoped question. See exactly what backs the answer.</p>
      </header>

      <TenantPicker tenants={tenants} selected={selectedTenant} onSelect={setSelectedTenant} />

      {selectedTenant && (
        <>
          <QueryPanel onSubmit={handleAsk} isLoading={isLoading} />
          {error && <p className="error">{error}</p>}
          {result && <AnswerCard result={result} />}

          <section className="audit-section">
            <h2>Audit trail</h2>
            <AuditTable entries={auditEntries} />
          </section>
        </>
      )}
    </div>
  )
}

export default App
