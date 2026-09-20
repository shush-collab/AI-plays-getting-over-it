using System;
using System.Diagnostics;
using System.IO;
using BepInEx;
using BepInEx.Unity.IL2CPP;
using Il2CppInterop.Runtime.InteropTypes.Arrays;
using UnityEngine;

namespace AIget.PhysicsWorker;

/// <summary>
/// Adds the runtime component after BepInEx has initialized IL2CPP interop.
/// </summary>
[BepInPlugin(PluginGuid, PluginName, PluginVersion)]
public sealed class PhysicsWorkerPlugin : BasePlugin
{
    public const string PluginGuid = "dev.aiget.physics-worker";
    public const string PluginName = "AIget Physics Worker";
    public const string PluginVersion = "0.1.0";

    public override void Load()
    {
        Log.LogInfo("Loading physics-only presentation controller.");
        AddComponent<PhysicsWorkerRuntime>();
    }
}

/// <summary>
/// Disables only visual and audio behaviours. It deliberately does not modify
/// colliders, Rigidbody2D, joints, MonoBehaviours, Animators, cameras, or game state.
/// </summary>
public sealed class PhysicsWorkerRuntime : MonoBehaviour
{
    private const int MaximumPasses = 3;
    private const float RetryIntervalSeconds = 1f;

    private readonly string _statusDirectory;
    private int _passes;
    private int _renderersDisabled;
    private int _audioSourcesDisabled;
    private int _lightsDisabled;
    private int _reflectionProbesDisabled;
    private bool _unusedAssetsUnloadRequested;

    public PhysicsWorkerRuntime(IntPtr pointer)
        : base(pointer)
    {
        _statusDirectory = Path.Combine(Paths.ConfigPath, "AIgetPhysicsWorker");
    }

    private void Start()
    {
        ApplyPresentationReduction();
        InvokeRepeating(nameof(ApplyPresentationReduction), RetryIntervalSeconds, RetryIntervalSeconds);
    }

    /// <summary>
    /// This is intentionally idempotent: later-created presentation objects are disabled
    /// during the short startup retry window without changing the physics simulation.
    /// </summary>
    public void ApplyPresentationReduction()
    {
        if (_passes >= MaximumPasses)
        {
            CancelInvoke(nameof(ApplyPresentationReduction));
            return;
        }

        _passes++;
        _renderersDisabled += DisableEnabledRenderers(UnityEngine.Object.FindObjectsOfType<Renderer>());
        _audioSourcesDisabled += DisableEnabledObjects(UnityEngine.Object.FindObjectsOfType<AudioSource>());
        _lightsDisabled += DisableEnabledObjects(UnityEngine.Object.FindObjectsOfType<Light>());
        _reflectionProbesDisabled += DisableEnabledObjects(UnityEngine.Object.FindObjectsOfType<ReflectionProbe>());
        if (!_unusedAssetsUnloadRequested)
        {
            Resources.UnloadUnusedAssets();
            _unusedAssetsUnloadRequested = true;
        }
        WriteStatus();

        if (_passes >= MaximumPasses)
        {
            CancelInvoke(nameof(ApplyPresentationReduction));
        }
    }

    private static int DisableEnabledObjects<T>(Il2CppArrayBase<T> objects)
        where T : Behaviour
    {
        var disabled = 0;
        foreach (var item in objects)
        {
            if (item is null || !item.enabled)
            {
                continue;
            }

            item.enabled = false;
            disabled++;
        }

        return disabled;
    }

    private static int DisableEnabledRenderers(Il2CppArrayBase<Renderer> renderers)
    {
        var disabled = 0;
        foreach (var renderer in renderers)
        {
            if (renderer is null || !renderer.enabled)
            {
                continue;
            }

            renderer.enabled = false;
            disabled++;
        }

        return disabled;
    }

    private void WriteStatus()
    {
        try
        {
            Directory.CreateDirectory(_statusDirectory);
            var workingSetBytes = Process.GetCurrentProcess().WorkingSet64;
            var json = $$"""
            {
              "plugin": "{{PhysicsWorkerPlugin.PluginGuid}}",
              "version": "{{PhysicsWorkerPlugin.PluginVersion}}",
              "passes": {{_passes}},
              "renderers_disabled": {{_renderersDisabled}},
              "audio_sources_disabled": {{_audioSourcesDisabled}},
              "lights_disabled": {{_lightsDisabled}},
              "reflection_probes_disabled": {{_reflectionProbesDisabled}},
              "unused_assets_unload_requested": {{_unusedAssetsUnloadRequested.ToString().ToLowerInvariant()}},
              "working_set_bytes": {{workingSetBytes}},
              "physics_preserved": true
            }
            """;
            var target = Path.Combine(_statusDirectory, "status.json");
            var temporary = target + ".tmp";
            File.WriteAllText(temporary, json + Environment.NewLine);
            if (File.Exists(target))
            {
                File.Delete(target);
            }

            File.Move(temporary, target);
        }
        catch (Exception exception)
        {
            UnityEngine.Debug.LogWarning($"AIget physics-worker could not write status: {exception.Message}");
        }
    }
}
