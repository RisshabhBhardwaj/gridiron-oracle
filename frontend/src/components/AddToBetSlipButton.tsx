/**
 * AddToBetSlipButton — the "+" button shown on player cards and detail pages.
 * Toggles the player between "in slip" and "remove from slip".
 */
import { useBetSlip, type BetSlipLeg } from '@/context/BetSlipContext'

interface AddToBetSlipButtonProps {
  leg: Omit<BetSlipLeg, 'id'>
  className?: string
  size?: 'sm' | 'md'
}

export function AddToBetSlipButton({ leg, size = 'sm' }: AddToBetSlipButtonProps) {
  const { addLeg, removeLeg, hasLeg } = useBetSlip()

  const id = `${leg.player_id}-${leg.stat}-${leg.week}`
  const inSlip = hasLeg(id)

  function handleClick(e: React.MouseEvent) {
    e.stopPropagation() // prevent card click navigation
    if (inSlip) {
      removeLeg(id)
    } else {
      addLeg({ ...leg, id })
    }
  }

  const isSmall = size === 'sm'

  return (
    <button
      onClick={handleClick}
      title={inSlip ? `Remove ${leg.player_name} from slip` : `Add ${leg.player_name} to bet slip`}
      className="flex items-center justify-center transition-all duration-150 rounded-lg font-bold select-none focus:outline-none focus-visible:ring-2 focus-visible:ring-oracle-blue"
      style={{
        width:  isSmall ? '28px' : '36px',
        height: isSmall ? '28px' : '36px',
        fontSize: isSmall ? '16px' : '20px',
        background: inSlip ? 'rgba(0,255,163,0.15)' : 'rgba(0,194,255,0.1)',
        border: `1px solid ${inSlip ? 'rgba(0,255,163,0.4)' : 'rgba(0,194,255,0.25)'}`,
        color: inSlip ? '#00FFA3' : '#00C2FF',
        boxShadow: inSlip ? '0 0 10px rgba(0,255,163,0.2)' : 'none',
        transform: inSlip ? 'rotate(45deg)' : 'rotate(0deg)', // × vs +
      }}
      aria-pressed={inSlip}
      aria-label={inSlip ? 'Remove from bet slip' : 'Add to bet slip'}
    >
      +
    </button>
  )
}
