using System.Collections.Generic;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Policies;
using Unity.MLAgents.Sensors;
using UnityEngine;
using UnityEngine.InputSystem;
using UnityEngine.InputSystem.Controls;
using UnityEngine.UI;

public sealed class TerritoryControlGame : MonoBehaviour
{
    public const int NeutralOwner = -1;
    public const int BlueTeam = 0;
    public const int OrangeTeam = 1;
    public const int ObservationSize = 21;

    [Header("Episode")]
    [SerializeField] private int maxEnvironmentSteps = 1800;
    [SerializeField] private bool enableKeyboardShortcuts = true;

    [Header("Arena")]
    [SerializeField] private int gridWidth = 25;
    [SerializeField] private int gridHeight = 17;
    [SerializeField] private int agentsPerTeam = 6;
    [SerializeField] private float tileSize = 1.15f;
    [SerializeField] private Color neutralColor = new(0.17f, 0.18f, 0.18f);
    [SerializeField] private Color blueColor = new(0.12f, 0.48f, 0.95f);
    [SerializeField] private Color orangeColor = new(1f, 0.42f, 0.12f);

    private readonly List<TerritoryPainterAgent> agents = new();
    private TileCell[,] tiles;
    private Material neutralMaterial;
    private Material blueMaterial;
    private Material orangeMaterial;
    private Text scoreText;
    private Text statusText;
    private int environmentStep;
    private int episodeIndex = 1;
    private bool paused;
    private bool endingEpisode;

    public int Width => gridWidth;
    public int Height => gridHeight;
    public float TileSize => tileSize;
    public bool IsRunning => !paused && !endingEpisode;

    [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.AfterSceneLoad)]
    private static void Bootstrap()
    {
        if (FindFirstObjectByType<TerritoryControlGame>() == null)
        {
            new GameObject("ML-Agents Territory Environment").AddComponent<TerritoryControlGame>();
        }
    }

    private void Awake()
    {
        Random.InitState(System.DateTime.UtcNow.Millisecond);
        Academy.Instance.AutomaticSteppingEnabled = true;

        CreateMaterials();
        CreateArena();
        CreateAgents();
        CreateLightingAndCamera();
        CreateHud();
        ResetEnvironment();
    }

    private void FixedUpdate()
    {
        HandleKeyboard();

        if (!IsRunning)
        {
            UpdateHud();
            return;
        }

        environmentStep++;
        GiveLivingRewards();

        if (environmentStep >= maxEnvironmentSteps)
        {
            FinishEpisode();
        }

        UpdateHud();
    }

    public Vector3 GridToWorld(Vector2Int cell)
    {
        float x = (cell.x - (gridWidth - 1) * 0.5f) * tileSize;
        float z = (cell.y - (gridHeight - 1) * 0.5f) * tileSize;
        return new Vector3(x, 0f, z);
    }

    public Vector2Int WorldToGrid(Vector3 worldPosition)
    {
        int x = Mathf.RoundToInt(worldPosition.x / tileSize + (gridWidth - 1) * 0.5f);
        int y = Mathf.RoundToInt(worldPosition.z / tileSize + (gridHeight - 1) * 0.5f);
        return new Vector2Int(Mathf.Clamp(x, 0, gridWidth - 1), Mathf.Clamp(y, 0, gridHeight - 1));
    }

    public int GetOwner(Vector2Int cell)
    {
        return IsInBounds(cell) ? tiles[cell.x, cell.y].Owner : NeutralOwner;
    }

    public bool PaintTile(Vector3 worldPosition, int teamId, out int previousOwner)
    {
        Vector2Int cell = WorldToGrid(worldPosition);
        previousOwner = NeutralOwner;

        if (!IsInBounds(cell))
        {
            return false;
        }

        TileCell tile = tiles[cell.x, cell.y];
        previousOwner = tile.Owner;

        if (previousOwner == teamId)
        {
            return false;
        }

        tile.SetOwner(teamId, GetTeamMaterial(teamId));
        return true;
    }

    public void AddObservationsFor(TerritoryPainterAgent agent, VectorSensor sensor)
    {
        Vector2Int cell = WorldToGrid(agent.transform.position);
        TerritoryPainterAgent nearestEnemy = FindNearestEnemy(agent, 12f);
        TerritoryPainterAgent nearestFriend = FindNearestFriend(agent, 12f);
        Vector3 toEnemy = nearestEnemy == null ? Vector3.zero : nearestEnemy.transform.position - agent.transform.position;
        Vector3 toFriend = nearestFriend == null ? Vector3.zero : nearestFriend.transform.position - agent.transform.position;
        Vector3 toEnemyBase = GetSpawnPoint(1 - agent.TeamId) - agent.transform.position;
        Vector3 toOwnBase = GetSpawnPoint(agent.TeamId) - agent.transform.position;
        float teamSide = agent.TeamId == BlueTeam ? 1f : -1f;

        sensor.AddObservation(teamSide * NormalizeCoord(cell.x, gridWidth));
        sensor.AddObservation(NormalizeCoord(cell.y, gridHeight));
        sensor.AddObservation(OwnerSignal(GetOwner(cell), agent.TeamId));

        Vector2Int[] offsets =
        {
            new(0, 1), new(1, 1), new(1, 0), new(1, -1),
            new(0, -1), new(-1, -1), new(-1, 0), new(-1, 1),
        };

        foreach (Vector2Int offset in offsets)
        {
            sensor.AddObservation(OwnerSignal(GetOwner(cell + offset), agent.TeamId));
        }

        sensor.AddObservation(Mathf.Clamp(toEnemy.x / (tileSize * 8f), -1f, 1f));
        sensor.AddObservation(Mathf.Clamp(toEnemy.z / (tileSize * 8f), -1f, 1f));
        sensor.AddObservation(nearestEnemy == null ? 1f : Mathf.Clamp01(toEnemy.magnitude / (tileSize * 12f)));
        sensor.AddObservation(Mathf.Clamp(toFriend.x / (tileSize * 8f), -1f, 1f));
        sensor.AddObservation(Mathf.Clamp(toFriend.z / (tileSize * 8f), -1f, 1f));
        sensor.AddObservation(nearestFriend == null ? 1f : Mathf.Clamp01(toFriend.magnitude / (tileSize * 12f)));
        sensor.AddObservation(Mathf.Clamp(toEnemyBase.x / (tileSize * gridWidth), -1f, 1f));
        sensor.AddObservation(Mathf.Clamp(toEnemyBase.z / (tileSize * gridHeight), -1f, 1f));
        sensor.AddObservation(Mathf.Clamp(toOwnBase.x / (tileSize * gridWidth), -1f, 1f));
        sensor.AddObservation(Mathf.Clamp(toOwnBase.z / (tileSize * gridHeight), -1f, 1f));
    }

    public Vector3 ClampAgentPosition(Vector3 worldPosition)
    {
        Vector2Int cell = WorldToGrid(worldPosition);
        Vector3 center = GridToWorld(cell);
        return new Vector3(
            Mathf.Clamp(worldPosition.x, center.x - tileSize * 0.45f, center.x + tileSize * 0.45f),
            0.35f,
            Mathf.Clamp(worldPosition.z, center.z - tileSize * 0.45f, center.z + tileSize * 0.45f));
    }

    public Vector3 GetSpawnPoint(int teamId)
    {
        int x = teamId == BlueTeam ? 1 : gridWidth - 2;
        int y = gridHeight / 2;
        Vector3 spawn = GridToWorld(new Vector2Int(x, y));
        spawn.y = 0.35f;
        return spawn;
    }

    public TerritoryPainterAgent FindNearestEnemy(TerritoryPainterAgent seeker, float maxDistance)
    {
        return FindNearestAgent(seeker, maxDistance, false);
    }

    public TerritoryPainterAgent FindNearestFriend(TerritoryPainterAgent seeker, float maxDistance)
    {
        return FindNearestAgent(seeker, maxDistance, true);
    }

    public void ResolveTag(TerritoryPainterAgent tagger, TerritoryPainterAgent tagged)
    {
        tagger.AddReward(0.35f);
        tagged.AddReward(-0.2f);
        tagged.Respawn();
    }

    public Color GetTeamColor(int teamId)
    {
        return teamId == BlueTeam ? blueColor : orangeColor;
    }

    private void ResetEnvironment()
    {
        endingEpisode = false;
        environmentStep = 0;

        for (int x = 0; x < gridWidth; x++)
        {
            for (int y = 0; y < gridHeight; y++)
            {
                tiles[x, y].SetOwner(NeutralOwner, neutralMaterial);
            }
        }

        PaintBaseZone(BlueTeam);
        PaintBaseZone(OrangeTeam);

        foreach (TerritoryPainterAgent agent in agents)
        {
            agent.ResetForEpisode(GetSpawnPoint(agent.TeamId));
        }
    }

    private void FinishEpisode()
    {
        if (endingEpisode)
        {
            return;
        }

        endingEpisode = true;
        CountTiles(out int blueTiles, out int orangeTiles, out _);
        int diff = blueTiles - orangeTiles;
        float normalizedDiff = Mathf.Clamp(diff / (float)(gridWidth * gridHeight), -1f, 1f);

        foreach (TerritoryPainterAgent agent in agents)
        {
            float teamResult = agent.TeamId == BlueTeam ? normalizedDiff : -normalizedDiff;
            agent.AddReward(teamResult);
            agent.EndEpisode();
        }

        episodeIndex++;
        ResetEnvironment();
    }

    private void GiveLivingRewards()
    {
        foreach (TerritoryPainterAgent agent in agents)
        {
            agent.AddReward(-0.0005f);
        }
    }

    private void HandleKeyboard()
    {
        if (!enableKeyboardShortcuts || Keyboard.current == null)
        {
            return;
        }

        if (Keyboard.current.spaceKey.wasPressedThisFrame)
        {
            paused = !paused;
        }

        if (Keyboard.current.rKey.wasPressedThisFrame)
        {
            ResetEnvironment();
        }

        if (Keyboard.current.nKey.wasPressedThisFrame)
        {
            FinishEpisode();
        }
    }

    private void CreateMaterials()
    {
        neutralMaterial = CreateMaterial(neutralColor);
        blueMaterial = CreateMaterial(blueColor);
        orangeMaterial = CreateMaterial(orangeColor);
    }

    private static Material CreateMaterial(Color color)
    {
        Material material = new(Shader.Find("Universal Render Pipeline/Lit"));
        material.color = color;
        return material;
    }

    private void CreateArena()
    {
        tiles = new TileCell[gridWidth, gridHeight];
        Transform arenaRoot = new GameObject("Paint Arena").transform;
        arenaRoot.SetParent(transform);

        for (int x = 0; x < gridWidth; x++)
        {
            for (int y = 0; y < gridHeight; y++)
            {
                GameObject tileObject = GameObject.CreatePrimitive(PrimitiveType.Cube);
                tileObject.name = $"Tile {x},{y}";
                tileObject.transform.SetParent(arenaRoot);
                tileObject.transform.position = GridToWorld(new Vector2Int(x, y));
                tileObject.transform.localScale = new Vector3(tileSize * 0.95f, 0.08f, tileSize * 0.95f);

                TileCell tile = tileObject.AddComponent<TileCell>();
                tile.SetOwner(NeutralOwner, neutralMaterial);
                tiles[x, y] = tile;
            }
        }
    }

    private void CreateAgents()
    {
        Transform agentRoot = new GameObject("ML Territory Agents").transform;
        agentRoot.SetParent(transform);

        for (int team = 0; team < 2; team++)
        {
            for (int i = 0; i < agentsPerTeam; i++)
            {
                GameObject agentObject = GameObject.CreatePrimitive(PrimitiveType.Capsule);
                agentObject.SetActive(false);
                agentObject.name = $"{(team == BlueTeam ? "Blue" : "Orange")} ML Agent {i + 1}";
                agentObject.transform.SetParent(agentRoot);
                agentObject.transform.localScale = new Vector3(0.42f, 0.42f, 0.42f);
                agentObject.transform.position = GetSpawnPoint(team);

                Renderer renderer = agentObject.GetComponent<Renderer>();
                renderer.sharedMaterial = CreateMaterial(GetTeamColor(team));

                TerritoryPainterAgent agent = agentObject.AddComponent<TerritoryPainterAgent>();
                agent.Initialize(this, team);

                BehaviorParameters behavior = agentObject.GetComponent<BehaviorParameters>();
                if (behavior == null)
                {
                    behavior = agentObject.AddComponent<BehaviorParameters>();
                }

                behavior.BehaviorName = "TerritoryPainter";
                behavior.TeamId = team;
                behavior.BehaviorType = BehaviorType.Default;
                behavior.BrainParameters.VectorObservationSize = ObservationSize;
                behavior.BrainParameters.ActionSpec = ActionSpec.MakeContinuous(2);

                DecisionRequester requester = agentObject.AddComponent<DecisionRequester>();
                requester.DecisionPeriod = 1;
                requester.TakeActionsBetweenDecisions = true;

                agents.Add(agent);
                agentObject.SetActive(true);
            }
        }
    }

    private void CreateLightingAndCamera()
    {
        if (FindFirstObjectByType<Light>() == null)
        {
            GameObject lightObject = new("Sun");
            Light light = lightObject.AddComponent<Light>();
            light.type = LightType.Directional;
            light.intensity = 2.2f;
            lightObject.transform.rotation = Quaternion.Euler(55f, -35f, 0f);
        }

        Camera camera = Camera.main;
        if (camera == null)
        {
            GameObject cameraObject = new("Main Camera");
            cameraObject.tag = "MainCamera";
            camera = cameraObject.AddComponent<Camera>();
        }

        camera.transform.position = new Vector3(0f, 18f, -17f);
        camera.transform.rotation = Quaternion.Euler(58f, 0f, 0f);
        camera.orthographic = true;
        camera.orthographicSize = Mathf.Max(gridWidth, gridHeight) * 0.62f;
        camera.clearFlags = CameraClearFlags.SolidColor;
        camera.backgroundColor = new Color(0.06f, 0.07f, 0.08f);
    }

    private void CreateHud()
    {
        Canvas canvas = new GameObject("ML-Agents HUD").AddComponent<Canvas>();
        canvas.transform.SetParent(transform);
        canvas.renderMode = RenderMode.ScreenSpaceOverlay;
        canvas.gameObject.AddComponent<CanvasScaler>();
        canvas.gameObject.AddComponent<GraphicRaycaster>();

        scoreText = CreateText(canvas.transform, "Score", new Vector2(18f, -18f), TextAnchor.UpperLeft, 25);
        statusText = CreateText(canvas.transform, "Status", new Vector2(-18f, -18f), TextAnchor.UpperRight, 23);
    }

    private static Text CreateText(Transform parent, string name, Vector2 anchoredPosition, TextAnchor anchor, int fontSize)
    {
        Text text = new GameObject(name).AddComponent<Text>();
        text.transform.SetParent(parent);
        text.font = Resources.GetBuiltinResource<Font>("LegacyRuntime.ttf");
        text.fontSize = fontSize;
        text.alignment = anchor;
        text.color = Color.white;

        RectTransform rect = text.GetComponent<RectTransform>();
        rect.anchorMin = AnchorFor(anchor);
        rect.anchorMax = AnchorFor(anchor);
        rect.pivot = AnchorFor(anchor);
        rect.anchoredPosition = anchoredPosition;
        rect.sizeDelta = new Vector2(560f, 120f);
        return text;
    }

    private static Vector2 AnchorFor(TextAnchor anchor)
    {
        return anchor switch
        {
            TextAnchor.UpperLeft => new Vector2(0f, 1f),
            TextAnchor.UpperRight => new Vector2(1f, 1f),
            _ => new Vector2(0.5f, 0.5f),
        };
    }

    private void UpdateHud()
    {
        CountTiles(out int blueTiles, out int orangeTiles, out int neutralTiles);
        int paintedTiles = Mathf.Max(1, gridWidth * gridHeight - neutralTiles);
        float blueShare = blueTiles / (float)paintedTiles;
        float orangeShare = orangeTiles / (float)paintedTiles;
        int remainingSteps = Mathf.Max(0, maxEnvironmentSteps - environmentStep);

        scoreText.text = $"EPISODE {episodeIndex}\nBLUE {blueTiles} ({blueShare:P0})\nORANGE {orangeTiles} ({orangeShare:P0})";
        statusText.text = paused
            ? "PAUSED"
            : $"ML-AGENTS\nSTEPS {remainingSteps}\nR reset  N end";
    }

    private void PaintBaseZone(int teamId)
    {
        Vector2Int center = WorldToGrid(GetSpawnPoint(teamId));
        for (int x = Mathf.Max(0, center.x - 1); x <= Mathf.Min(gridWidth - 1, center.x + 1); x++)
        {
            for (int y = Mathf.Max(0, center.y - 2); y <= Mathf.Min(gridHeight - 1, center.y + 2); y++)
            {
                tiles[x, y].SetOwner(teamId, GetTeamMaterial(teamId));
            }
        }
    }

    private TerritoryPainterAgent FindNearestAgent(TerritoryPainterAgent seeker, float maxDistance, bool sameTeam)
    {
        TerritoryPainterAgent nearest = null;
        float bestSqrDistance = maxDistance * maxDistance;

        foreach (TerritoryPainterAgent other in agents)
        {
            if (other == seeker || other.IsRespawning || (other.TeamId == seeker.TeamId) != sameTeam)
            {
                continue;
            }

            float sqrDistance = (other.transform.position - seeker.transform.position).sqrMagnitude;
            if (sqrDistance < bestSqrDistance)
            {
                bestSqrDistance = sqrDistance;
                nearest = other;
            }
        }

        return nearest;
    }

    private Material GetTeamMaterial(int teamId)
    {
        return teamId == BlueTeam ? blueMaterial : orangeMaterial;
    }

    private bool IsInBounds(Vector2Int cell)
    {
        return cell.x >= 0 && cell.x < gridWidth && cell.y >= 0 && cell.y < gridHeight;
    }

    private static float NormalizeCoord(int value, int size)
    {
        return size <= 1 ? 0f : (value / (float)(size - 1)) * 2f - 1f;
    }

    private static float OwnerSignal(int owner, int teamId)
    {
        if (owner == NeutralOwner)
        {
            return 0f;
        }

        return owner == teamId ? 1f : -1f;
    }

    private void CountTiles(out int blueTiles, out int orangeTiles, out int neutralTiles)
    {
        blueTiles = 0;
        orangeTiles = 0;
        neutralTiles = 0;

        for (int x = 0; x < gridWidth; x++)
        {
            for (int y = 0; y < gridHeight; y++)
            {
                switch (tiles[x, y].Owner)
                {
                    case BlueTeam:
                        blueTiles++;
                        break;
                    case OrangeTeam:
                        orangeTiles++;
                        break;
                    default:
                        neutralTiles++;
                        break;
                }
            }
        }
    }
}

public sealed class TerritoryPainterAgent : Agent
{
    [SerializeField] private float moveSpeed = 3.8f;
    [SerializeField] private float paintInterval = 0.12f;
    [SerializeField] private float tagDistance = 0.55f;
    [SerializeField] private float respawnSeconds = 1.3f;

    private TerritoryControlGame game;
    private Renderer bodyRenderer;
    private Vector3 actionDirection;
    private float nextPaintAt;
    private float respawnUntil;

    public int TeamId { get; private set; }
    public bool IsRespawning => Time.time < respawnUntil;

    public void Initialize(TerritoryControlGame owner, int teamId)
    {
        game = owner;
        TeamId = teamId;
        bodyRenderer = GetComponent<Renderer>();
    }

    public override void OnEpisodeBegin()
    {
        actionDirection = Random.insideUnitSphere.WithY(0f).normalized;
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        if (game == null)
        {
            AddEmptyObservations(sensor);
            return;
        }

        game.AddObservationsFor(this, sensor);
    }

    public override void OnActionReceived(ActionBuffers actions)
    {
        if (game == null || !game.IsRunning || IsRespawning)
        {
            return;
        }

        actionDirection = new Vector3(
            Mathf.Clamp(actions.ContinuousActions[0], -1f, 1f),
            0f,
            Mathf.Clamp(actions.ContinuousActions[1], -1f, 1f));

        if (actionDirection.sqrMagnitude < 0.03f)
        {
            AddReward(-0.002f);
            return;
        }

        actionDirection.Normalize();
        Move();
        Paint();
        TryTagEnemy();
    }

    public override void Heuristic(in ActionBuffers actionsOut)
    {
        ActionSegment<float> continuous = actionsOut.ContinuousActions;
        Vector2 input = Vector2.zero;

        if (Keyboard.current != null)
        {
            input.x = ReadAxis(Keyboard.current.dKey, Keyboard.current.aKey);
            input.y = ReadAxis(Keyboard.current.wKey, Keyboard.current.sKey);
        }

        continuous[0] = input.x;
        continuous[1] = input.y;
    }

    public void ResetForEpisode(Vector3 spawnPoint)
    {
        transform.position = spawnPoint + Random.insideUnitSphere.WithY(0f) * 0.35f;
        actionDirection = Random.insideUnitSphere.WithY(0f).normalized;
        respawnUntil = 0f;
        nextPaintAt = 0f;

        if (bodyRenderer != null)
        {
            bodyRenderer.enabled = true;
        }
    }

    public void Respawn()
    {
        transform.position = game.GetSpawnPoint(TeamId);
        actionDirection = Random.insideUnitSphere.WithY(0f).normalized;
        respawnUntil = Time.time + respawnSeconds;
    }

    private void Update()
    {
        if (bodyRenderer == null)
        {
            return;
        }

        bodyRenderer.enabled = !IsRespawning || Mathf.FloorToInt(Time.time * 10f) % 2 == 0;
    }

    private void Move()
    {
        int currentOwner = game.GetOwner(game.WorldToGrid(transform.position));
        float territoryMultiplier = currentOwner == TeamId ? 1.12f : currentOwner == TerritoryControlGame.NeutralOwner ? 1f : 0.78f;
        Vector3 movement = actionDirection * moveSpeed * territoryMultiplier * Time.fixedDeltaTime;
        transform.position = game.ClampAgentPosition(transform.position + movement);
        transform.rotation = Quaternion.Slerp(transform.rotation, Quaternion.LookRotation(actionDirection), Time.fixedDeltaTime * 12f);
    }

    private void Paint()
    {
        if (Time.time < nextPaintAt)
        {
            return;
        }

        nextPaintAt = Time.time + paintInterval;
        if (game.PaintTile(transform.position, TeamId, out int previousOwner))
        {
            AddReward(previousOwner == TerritoryControlGame.NeutralOwner ? 0.08f : 0.18f);
        }
    }

    private void TryTagEnemy()
    {
        TerritoryPainterAgent enemy = game.FindNearestEnemy(this, tagDistance);
        if (enemy == null)
        {
            return;
        }

        int enemyOwner = game.GetOwner(game.WorldToGrid(enemy.transform.position));
        if (enemyOwner == TeamId)
        {
            game.ResolveTag(this, enemy);
        }
    }

    private static float ReadAxis(KeyControl positive, KeyControl negative)
    {
        float value = 0f;

        if (positive.isPressed)
        {
            value += 1f;
        }

        if (negative.isPressed)
        {
            value -= 1f;
        }

        return value;
    }

    private static void AddEmptyObservations(VectorSensor sensor)
    {
        for (int i = 0; i < TerritoryControlGame.ObservationSize; i++)
        {
            sensor.AddObservation(0f);
        }
    }
}

public sealed class TileCell : MonoBehaviour
{
    private Renderer tileRenderer;

    public int Owner { get; private set; } = TerritoryControlGame.NeutralOwner;

    public void SetOwner(int owner, Material material)
    {
        Owner = owner;

        if (tileRenderer == null)
        {
            tileRenderer = GetComponent<Renderer>();
        }

        tileRenderer.sharedMaterial = material;
    }
}

public static class VectorExtensions
{
    public static Vector3 WithY(this Vector3 vector, float y)
    {
        vector.y = y;
        return vector;
    }
}
