export interface TenantSummary {
  tenant_id: string
  display_name: string
}

export interface Citation {
  assertion: string
  source_id: string
  document_id: string
  text: string
}

export interface QueryResponse {
  request_id: string
  tenant_id: string
  answer: string | null
  rung: number
  rung_name: string
  citations: Citation[]
  unsupported_assertions: string[]
}

export interface AuditEntry {
  request_id: string
  tenant_id: string
  request_text: string
  answer: string | null
  rung: number | null
  rung_name: string | null
  created_at: string
}

const BASE_URL = 'http://localhost:8000'

async function asJson<T>(response: Response): Promise<T> {
  if (!response.ok) {
    const detail = await response.text()
    throw new Error(`${response.status} ${response.statusText}: ${detail}`)
  }
  return response.json() as Promise<T>
}

export function listTenants(): Promise<TenantSummary[]> {
  return fetch(`${BASE_URL}/tenants`).then((response) => asJson<TenantSummary[]>(response))
}

export function queryTenant(tenantId: string, userRequest: string): Promise<QueryResponse> {
  return fetch(`${BASE_URL}/tenants/${tenantId}/query`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ user_request: userRequest }),
  }).then((response) => asJson<QueryResponse>(response))
}

export function listAudit(tenantId: string): Promise<AuditEntry[]> {
  return fetch(`${BASE_URL}/tenants/${tenantId}/audit`).then((response) =>
    asJson<AuditEntry[]>(response),
  )
}
