package ai_test

import (
	"context"
	"errors"
	"slices"
	"testing"

	"github.com/lace-ai/gai/ai"
	"github.com/lace-ai/gai/testutil/mocks"
)

func TestModelRepository(t *testing.T) {
	repo := ai.NewModelRepository(nil)

	// Test registering a provider
	provider := &mocks.MockProvider{ProviderName: "mock"}
	err := repo.RegisterProvider(context.Background(), provider)
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}

	// Test registering the same provider again
	err = repo.RegisterProvider(context.Background(), provider)
	if err == nil {
		t.Fatalf("expected error when registering duplicate provider, got nil")
	}
}

func TestModelRepositoryRejectsNilProvider(t *testing.T) {
	repo := ai.NewModelRepository(nil)

	err := repo.RegisterProvider(context.Background(), nil)
	if !errors.Is(err, ai.ErrNilProvider) {
		t.Fatalf("expected ErrNilProvider, got %v", err)
	}
}

func TestModelRepositoryListModelDescriptors(t *testing.T) {
	repo := ai.NewModelRepository(nil)
	provider := &mocks.MockProvider{ProviderName: "mock", Models: map[string]ai.Model{
		"legacy":    &mocks.MockModel{ModelName: "legacy"},
		"described": &describedModel{MockModel: mocks.MockModel{ModelName: "described"}, descriptor: ai.ModelDescriptor{NativeMessages: ai.FeatureSupportSupported, ToolChoiceModes: []ai.ToolChoiceMode{ai.ToolChoiceAuto}}},
	}}
	if err := repo.RegisterProvider(context.Background(), provider); err != nil {
		t.Fatal(err)
	}

	descriptors, err := repo.ListModelDescriptors(context.Background())
	if err != nil {
		t.Fatalf("ListModelDescriptors: %v", err)
	}
	if len(descriptors) != 1 || descriptors[0].Provider != "mock" || descriptors[0].Model != "described" || descriptors[0].NativeMessages != ai.FeatureSupportSupported {
		t.Fatalf("descriptors = %#v", descriptors)
	}
	descriptors[0].ToolChoiceModes[0] = ai.ToolChoiceNone
	again, err := repo.ListModelDescriptors(context.Background())
	if err != nil || again[0].ToolChoiceModes[0] != ai.ToolChoiceAuto {
		t.Fatalf("aggregation did not return independent descriptor copies: %#v, %v", again, err)
	}
}

func TestModelRepositoryPrefersContextAwareModelCatalog(t *testing.T) {
	repo := ai.NewModelRepository(nil)
	provider := &catalogProvider{
		MockProvider: mocks.MockProvider{ProviderName: "catalog"},
		descriptors: []ai.ModelDescriptor{{
			Model:           "dynamic",
			NativeMessages:  ai.FeatureSupportSupported,
			ToolChoiceModes: []ai.ToolChoiceMode{ai.ToolChoiceAuto},
		}},
	}
	if err := repo.RegisterProvider(context.Background(), provider); err != nil {
		t.Fatal(err)
	}

	type contextKey string
	ctx := context.WithValue(context.Background(), contextKey("key"), "value")
	descriptors, err := repo.ListModelDescriptors(ctx)
	if err != nil {
		t.Fatalf("ListModelDescriptors: %v", err)
	}
	if provider.ctx != ctx {
		t.Fatal("repository did not propagate the caller context")
	}
	if provider.legacyCalls != 0 {
		t.Fatalf("legacy provider path called %d times", provider.legacyCalls)
	}
	names, err := repo.ListModels(ctx)
	if err != nil || !slices.Equal(names, []string{"catalog:dynamic"}) {
		t.Fatalf("ListModels = %v, %v", names, err)
	}
	if provider.ctx != ctx || provider.legacyCalls != 0 {
		t.Fatal("ListModels must also prefer the caller-context catalog")
	}
	if len(descriptors) != 1 || descriptors[0].Provider != "catalog" || descriptors[0].Model != "dynamic" {
		t.Fatalf("descriptors = %#v", descriptors)
	}

	descriptors[0].ToolChoiceModes[0] = ai.ToolChoiceNone
	again, err := repo.ListModelDescriptors(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if again[0].ToolChoiceModes[0] != ai.ToolChoiceAuto {
		t.Fatalf("catalog result was not copy isolated: %#v", again)
	}
}

type describedModel struct {
	mocks.MockModel
	descriptor ai.ModelDescriptor
}

func (m *describedModel) Descriptor() ai.ModelDescriptor { return m.descriptor }

type catalogProvider struct {
	mocks.MockProvider
	ctx         context.Context
	descriptors []ai.ModelDescriptor
	legacyCalls int
}

func (p *catalogProvider) ListModelDescriptors(ctx context.Context) ([]ai.ModelDescriptor, error) {
	p.ctx = ctx
	out := make([]ai.ModelDescriptor, len(p.descriptors))
	for i := range p.descriptors {
		out[i] = p.descriptors[i].Copy()
	}
	return out, nil
}

func (p *catalogProvider) ListModels() ([]string, error) {
	p.legacyCalls++
	return nil, errors.New("legacy ListModels must not be called")
}

func (p *catalogProvider) Model(string) (ai.Model, error) {
	p.legacyCalls++
	return nil, errors.New("legacy Model must not be called")
}

// minimalProvider deliberately has neither validation nor discovery methods.
type minimalProvider struct {
	name  string
	model func(string) (ai.Model, error)
}

func (p *minimalProvider) Name() string { return p.name }
func (p *minimalProvider) Model(name string) (ai.Model, error) {
	return p.model(name)
}

type validatingProvider struct {
	minimalProvider
	validate func() error
}

func (p *validatingProvider) Validate() error { return p.validate() }

type namesOnlyProvider struct {
	minimalProvider
	list func() ([]string, error)
}

func (p *namesOnlyProvider) ListModels() ([]string, error) { return p.list() }

type catalogOnlyProvider struct {
	minimalProvider
	list func(context.Context) ([]ai.ModelDescriptor, error)
}

func (p *catalogOnlyProvider) ListModelDescriptors(ctx context.Context) ([]ai.ModelDescriptor, error) {
	return p.list(ctx)
}

func TestModelRepositoryMinimalProvider(t *testing.T) {
	repo := ai.NewModelRepository(nil)
	want := &mocks.MockModel{ModelName: "known-model"}
	provider := &minimalProvider{name: "minimal", model: func(name string) (ai.Model, error) {
		if name != "known-model" {
			return nil, ai.ErrModelNotFound
		}
		return want, nil
	}}
	if err := repo.RegisterProvider(t.Context(), provider); err != nil {
		t.Fatal(err)
	}
	got, err := repo.GetModel(t.Context(), "minimal", "known-model")
	if err != nil || got != want {
		t.Fatalf("GetModel = %v, %v; want registered model", got, err)
	}
	if _, err := repo.GetModel(t.Context(), "minimal", "missing"); !errors.Is(err, ai.ErrModelNotFound) {
		t.Fatalf("missing model error = %v", err)
	}
	if err := repo.RegisterProvider(t.Context(), provider); !errors.Is(err, ai.ErrProviderAlreadyExists) {
		t.Fatalf("duplicate registration error = %v", err)
	}
}

func TestModelRepositoryLookupDoesNotDiscover(t *testing.T) {
	want := &mocks.MockModel{ModelName: "known"}
	provider := &catalogOnlyProvider{
		minimalProvider: minimalProvider{name: "catalog", model: func(string) (ai.Model, error) { return want, nil }},
		list: func(context.Context) ([]ai.ModelDescriptor, error) {
			t.Fatal("named lookup must not call discovery")
			return nil, nil
		},
	}
	repo := ai.NewModelRepository(nil)
	if err := repo.RegisterProvider(t.Context(), provider); err != nil {
		t.Fatal(err)
	}
	if got, err := repo.GetModel(t.Context(), "catalog", "known"); err != nil || got != want {
		t.Fatalf("GetModel = %v, %v", got, err)
	}
}

func TestModelRepositoryOptionalValidation(t *testing.T) {
	repo := ai.NewModelRepository(nil)
	wantErr := errors.New("invalid credentials")
	calls := 0
	provider := &validatingProvider{
		minimalProvider: minimalProvider{name: "validated"},
		validate:        func() error { calls++; return wantErr },
	}
	if err := repo.RegisterProvider(t.Context(), provider); !errors.Is(err, wantErr) {
		t.Fatalf("RegisterProvider = %v, want validation error", err)
	}
	if _, err := repo.GetModel(t.Context(), "validated", "any"); !errors.Is(err, ai.ErrProviderNotFound) {
		t.Fatalf("invalid provider was registered: %v", err)
	}
	provider.validate = func() error { calls++; return nil }
	if err := repo.RegisterProvider(t.Context(), provider); err != nil {
		t.Fatal(err)
	}
	if calls != 2 {
		t.Fatalf("Validate called %d times, want 2", calls)
	}
}

func TestModelRepositoryValidatesIdentityBeforeOptionalValidation(t *testing.T) {
	for _, name := range []string{"", " ", "\t\n"} {
		t.Run(name, func(t *testing.T) {
			provider := &validatingProvider{
				minimalProvider: minimalProvider{name: name},
				validate: func() error {
					t.Fatal("invalid provider name must be rejected before optional validation")
					return nil
				},
			}
			repo := ai.NewModelRepository(nil)
			if err := repo.RegisterProvider(t.Context(), provider); !errors.Is(err, ai.ErrProviderInvalid) {
				t.Fatalf("RegisterProvider = %v, want ErrProviderInvalid", err)
			}
		})
	}
	var provider *minimalProvider
	if err := ai.NewModelRepository(nil).RegisterProvider(t.Context(), provider); !errors.Is(err, ai.ErrNilProvider) {
		t.Fatalf("typed nil RegisterProvider = %v, want ErrNilProvider", err)
	}
}

// Normalize both listing APIs for tests of their shared error and context
// contracts. A nil result is preserved so partial-result regressions are caught.
var repositoryListings = []struct {
	name string
	list func(context.Context, *ai.ModelRepository) ([]string, error)
}{
	{"names", func(ctx context.Context, repo *ai.ModelRepository) ([]string, error) { return repo.ListModels(ctx) }},
	{"descriptors", func(ctx context.Context, repo *ai.ModelRepository) ([]string, error) {
		descriptors, err := repo.ListModelDescriptors(ctx)
		var names []string
		if descriptors != nil {
			names = make([]string, 0, len(descriptors))
		}
		for _, descriptor := range descriptors {
			names = append(names, descriptor.Provider+":"+descriptor.Model)
		}
		return names, err
	}},
}

func TestModelRepositoryDiscoveryIsAllOrError(t *testing.T) {
	for _, listing := range repositoryListings {
		t.Run(listing.name, func(t *testing.T) {
			repo := ai.NewModelRepository(nil)
			for _, provider := range []ai.Provider{
				&minimalProvider{name: "z-unsupported"},
				&minimalProvider{name: "b-unsupported"},
				&catalogOnlyProvider{
					minimalProvider: minimalProvider{name: "a-supported"},
					list: func(context.Context) ([]ai.ModelDescriptor, error) {
						return []ai.ModelDescriptor{{Model: "available"}}, nil
					},
				},
			} {
				if err := repo.RegisterProvider(t.Context(), provider); err != nil {
					t.Fatal(err)
				}
			}
			names, err := listing.list(t.Context(), repo)
			var discoveryErr *ai.UnsupportedModelDiscoveryError
			if !errors.Is(err, ai.ErrUnsupportedCapability) || !errors.As(err, &discoveryErr) {
				t.Fatalf("error = %v, want typed unsupported discovery", err)
			}
			if discoveryErr.Provider != "b-unsupported" {
				t.Fatalf("failed provider = %q, want deterministic first unsupported provider", discoveryErr.Provider)
			}
			if names != nil {
				t.Fatalf("partial result = %v, want nil", names)
			}
		})
	}
}

func TestModelRepositoryOptionalDiscovery(t *testing.T) {
	for _, listing := range repositoryListings {
		t.Run(listing.name, func(t *testing.T) {
			repo := ai.NewModelRepository(nil)
			var calls []string
			for _, name := range []string{"z-catalog", "a-names"} {
				base := minimalProvider{name: name, model: func(string) (ai.Model, error) {
					if name == "z-catalog" {
						t.Fatal("catalog discovery must not resolve model instances")
					}
					return &describedModel{}, nil
				}}
				var provider ai.Provider
				if name == "z-catalog" {
					provider = &catalogOnlyProvider{minimalProvider: base, list: func(ctx context.Context) ([]ai.ModelDescriptor, error) {
						if ctx != t.Context() {
							t.Fatal("caller context not passed to catalog")
						}
						calls = append(calls, name)
						return []ai.ModelDescriptor{{Provider: "ignored", Model: "second"}, {Model: "first"}, {}}, nil
					}}
				} else {
					provider = &namesOnlyProvider{minimalProvider: base, list: func() ([]string, error) {
						calls = append(calls, name)
						return []string{"second", "first"}, nil
					}}
				}
				if err := repo.RegisterProvider(t.Context(), provider); err != nil {
					t.Fatal(err)
				}
			}
			got, err := listing.list(t.Context(), repo)
			want := []string{"a-names:first", "a-names:second", "z-catalog:first", "z-catalog:second"}
			if err != nil || !slices.Equal(got, want) {
				t.Fatalf("listing = %v, %v; want %v", got, err, want)
			}
			if !slices.Equal(calls, []string{"a-names", "z-catalog"}) {
				t.Fatalf("provider discovery order = %v", calls)
			}
		})
	}
}

func TestModelRepositoryEmptyDiscoveryIsSupported(t *testing.T) {
	for _, listing := range repositoryListings {
		t.Run(listing.name, func(t *testing.T) {
			repo := ai.NewModelRepository(nil)
			for _, provider := range []ai.Provider{
				&namesOnlyProvider{minimalProvider: minimalProvider{name: "names"}, list: func() ([]string, error) { return nil, nil }},
				&catalogOnlyProvider{minimalProvider: minimalProvider{name: "catalog"}, list: func(context.Context) ([]ai.ModelDescriptor, error) { return nil, nil }},
			} {
				if err := repo.RegisterProvider(t.Context(), provider); err != nil {
					t.Fatal(err)
				}
			}
			if names, err := listing.list(t.Context(), repo); err != nil || len(names) != 0 {
				t.Fatalf("empty listing = %v, %v", names, err)
			}
		})
	}
}

func TestModelRepositoryDiscoveryCancellation(t *testing.T) {
	for _, listing := range repositoryListings {
		for _, capability := range []string{"catalog", "names"} {
			for _, before := range []bool{false, true} {
				name := listing.name + "/" + capability + "/during"
				if before {
					name = listing.name + "/" + capability + "/before"
				}
				t.Run(name, func(t *testing.T) {
					ctx, cancel := context.WithCancel(t.Context())
					defer cancel()
					calls := 0
					base := minimalProvider{name: "a-cancel", model: func(string) (ai.Model, error) {
						t.Fatal("cancellation must prevent fallback model lookup")
						return nil, nil
					}}
					var provider ai.Provider
					if capability == "catalog" {
						provider = &catalogOnlyProvider{minimalProvider: base, list: func(got context.Context) ([]ai.ModelDescriptor, error) {
							calls++
							if got != ctx {
								t.Fatal("caller context not propagated")
							}
							cancel()
							return []ai.ModelDescriptor{{Model: "discard"}}, nil
						}}
					} else {
						provider = &namesOnlyProvider{minimalProvider: base, list: func() ([]string, error) {
							calls++
							cancel()
							return []string{"discard"}, nil
						}}
					}
					repo := ai.NewModelRepository(nil)
					if err := repo.RegisterProvider(t.Context(), provider); err != nil {
						t.Fatal(err)
					}
					if err := repo.RegisterProvider(t.Context(), &minimalProvider{name: "z-next"}); err != nil {
						t.Fatal(err)
					}
					wantCalls := 1
					if before {
						cancel()
						wantCalls = 0
					}
					got, err := listing.list(ctx, repo)
					if !errors.Is(err, context.Canceled) || got != nil || calls != wantCalls {
						t.Fatalf("listing = %v, %v, calls = %d; want nil, Canceled, %d calls", got, err, calls, wantCalls)
					}
				})
			}
		}
	}
}

func TestModelRepositoryDescriptorLookupCancellation(t *testing.T) {
	ctx, cancel := context.WithCancel(t.Context())
	defer cancel()
	calls := 0
	provider := &namesOnlyProvider{
		minimalProvider: minimalProvider{name: "names", model: func(string) (ai.Model, error) {
			calls++
			cancel()
			return &describedModel{}, nil
		}},
		list: func() ([]string, error) { return []string{"first", "second"}, nil },
	}
	repo := ai.NewModelRepository(nil)
	if err := repo.RegisterProvider(t.Context(), provider); err != nil {
		t.Fatal(err)
	}
	got, err := repo.ListModelDescriptors(ctx)
	if !errors.Is(err, context.Canceled) || got != nil || calls != 1 {
		t.Fatalf("listing = %v, %v, lookups = %d; want nil, Canceled, 1 lookup", got, err, calls)
	}
}
