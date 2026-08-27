import type { TenantSummary } from '../api/client'

interface Props {
  tenants: TenantSummary[]
  selected: string | null
  onSelect: (tenantId: string) => void
}

export function TenantPicker({ tenants, selected, onSelect }: Props) {
  return (
    <div className="tenant-picker">
      {tenants.map((tenant) => (
        <button
          key={tenant.tenant_id}
          type="button"
          className={tenant.tenant_id === selected ? 'tenant-button active' : 'tenant-button'}
          onClick={() => onSelect(tenant.tenant_id)}
        >
          {tenant.display_name}
        </button>
      ))}
    </div>
  )
}
