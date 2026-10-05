package loop

import (
	"context"
	"fmt"
	"slices"
)

// ToolAction is an explicit authorization decision. A custom policy returning
// the zero value is invalid and fails closed.
type ToolAction string

const (
	ToolAllow           ToolAction = "allow"
	ToolDeny            ToolAction = "deny"
	ToolRequireApproval ToolAction = "require_approval"
)

// ToolDecision separates authorization from whether invocation later happened.
// Code is an application reason code; Reason must be safe for users and models.
type ToolDecision struct {
	Action ToolAction
	Code   string
	Reason string
}

// Validate rejects unspecified or unknown authorization actions.
func (d ToolDecision) Validate() error {
	switch d.Action {
	case ToolAllow, ToolDeny, ToolRequireApproval:
		return nil
	}
	return fmt.Errorf("%w: invalid action %q", ErrToolPolicy, d.Action)
}

// ToolPolicy authorizes a copied request. Policy errors terminate the run;
// expected denials and approval requirements are ordinary decisions.
type ToolPolicy interface {
	BeforeTool(context.Context, ToolPolicyInput) (ToolDecision, error)
}

// ToolPolicyFunc adapts a concurrency-safe authorization callback.
type ToolPolicyFunc func(context.Context, ToolPolicyInput) (ToolDecision, error)

// BeforeTool invokes the policy callback, rejecting a nil function.
func (f ToolPolicyFunc) BeforeTool(ctx context.Context, input ToolPolicyInput) (ToolDecision, error) {
	if f == nil {
		return ToolDecision{}, fmt.Errorf("%w: nil policy function", ErrToolPolicy)
	}
	return f(ctx, input)
}

// ToolPolicyRules implements name/effect rules without an application policy.
// Precedence: Deny, RequireApproval/ApprovalEffects, Allow, Default.
// An omitted Default denies unmatched calls when Allow is nonempty, otherwise
// allows them. Explicit Default overrides that fallback. Traits are declarations.
type ToolPolicyRules struct {
	Default         ToolAction
	Allow           []string
	Deny            []string
	RequireApproval []string
	ApprovalEffects []ToolEffect
}

type rulesPolicy struct{ rules ToolPolicyRules }

// NewToolPolicy validates and snapshots the supplied rules.
func NewToolPolicy(rules ToolPolicyRules) (ToolPolicy, error) {
	if rules.Default == "" {
		rules.Default = ToolAllow
		if len(rules.Allow) > 0 {
			rules.Default = ToolDeny
		}
	}
	if err := (ToolDecision{Action: rules.Default}).Validate(); err != nil {
		return nil, err
	}
	rules.Allow = slices.Clone(rules.Allow)
	rules.Deny = slices.Clone(rules.Deny)
	rules.RequireApproval = slices.Clone(rules.RequireApproval)
	rules.ApprovalEffects = slices.Clone(rules.ApprovalEffects)
	for _, effect := range rules.ApprovalEffects {
		if err := (ToolOptions{Traits: ToolTraits{Effect: effect}}).Validate(); err != nil {
			return nil, err
		}
	}
	return rulesPolicy{rules: rules}, nil
}
func (p rulesPolicy) BeforeTool(_ context.Context, input ToolPolicyInput) (ToolDecision, error) {
	action := p.rules.Default
	switch {
	case slices.Contains(p.rules.Deny, input.Call.Name):
		action = ToolDeny
	case slices.Contains(p.rules.RequireApproval, input.Call.Name) || slices.Contains(p.rules.ApprovalEffects, input.Traits.Effect):
		action = ToolRequireApproval
	case slices.Contains(p.rules.Allow, input.Call.Name):
		action = ToolAllow
	}
	return ToolDecision{Action: action}, nil
}

// ChainToolPolicies uses the most restrictive decision: deny > approval > allow.
// Every policy sees an independent snapshot; any policy failure terminates.
func ChainToolPolicies(policies ...ToolPolicy) (ToolPolicy, error) {
	for _, p := range policies {
		if nilImplementation(p) {
			return nil, fmt.Errorf("%w: nil policy", ErrToolPolicy)
		}
	}
	policies = slices.Clone(policies)
	return ToolPolicyFunc(func(ctx context.Context, input ToolPolicyInput) (ToolDecision, error) {
		decision := ToolDecision{Action: ToolAllow}
		haveDecision := false
		for _, policy := range policies {
			if err := ctx.Err(); err != nil {
				return ToolDecision{}, err
			}
			current, err := evaluateToolPolicy(ctx, policy, input)
			if err != nil {
				return ToolDecision{}, err
			}
			if !haveDecision || actionRank(current.Action) > actionRank(decision.Action) {
				haveDecision = true
				decision = current
			}
		}
		return decision, nil
	}), nil
}
func actionRank(action ToolAction) int {
	switch action {
	case ToolDeny:
		return 3
	case ToolRequireApproval:
		return 2
	case ToolAllow:
		return 1
	}
	return 0
}
func evaluateToolPolicy(ctx context.Context, policy ToolPolicy, input ToolPolicyInput) (decision ToolDecision, err error) {
	defer func() {
		if recover() != nil {
			decision = ToolDecision{}
			err = ErrToolPanic
		}
	}()
	input.Call = input.Call.Clone()
	decision, err = policy.BeforeTool(ctx, input)
	if err != nil {
		return ToolDecision{}, fmt.Errorf("%w: %w", ErrToolPolicy, err)
	}
	if err = decision.Validate(); err != nil {
		return ToolDecision{}, err
	}
	return decision, nil
}
